use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_lfm2::ModelWeights as Lfm2Model;
use candle_transformers::models::quantized_llama::ModelWeights as LlamaModel;

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub name: String,
    pub architecture: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub file_size: u64,
    pub chat_template: Option<String>,
}

pub enum ModelInner {
    Llama(LlamaModel),
    Lfm2(Lfm2Model),
}

pub struct Model {
    inner: ModelInner,
    metadata: GgufMetadata,
}

impl Model {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let file_size = std::fs::metadata(path)?.len();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let device = Device::Cpu;

        let mut file =
            File::open(path).with_context(|| format!("Failed to open model file: {:?}", path))?;

        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF file: {:?}", path))?;

        let metadata = Self::extract_metadata(&content, filename, file_size)?;

        let arch = metadata.architecture.as_str();
        tracing::info!(
            "Loading model: {} ({} layers, {} embedding dim, {} vocab, arch: {})",
            metadata.name,
            metadata.n_layer,
            metadata.n_embd,
            metadata.vocab_size,
            arch
        );

        let inner = if arch == "lfm2" {
            let weights = Lfm2Model::from_gguf(content, &mut file, &device)
                .with_context(|| "Failed to load LFM2 model weights from GGUF")?;
            ModelInner::Lfm2(weights)
        } else {
            let weights = LlamaModel::from_gguf(content, &mut file, &device)
                .with_context(|| "Failed to load LLaMA model weights from GGUF")?;
            ModelInner::Llama(weights)
        };

        Ok(Self { inner, metadata })
    }

    fn extract_metadata(
        content: &gguf_file::Content,
        filename: &str,
        file_size: u64,
    ) -> Result<GgufMetadata> {
        let md = &content.metadata;

        let arch: String = match md.get("general.architecture") {
            Some(v) => v
                .to_string()
                .map(|s| s.clone())
                .unwrap_or_else(|_| "llama".to_string()),
            None => "llama".to_string(),
        };

        let model_name: String = md
            .get("general.name")
            .and_then(|v| v.to_string().ok().map(|s| s.clone()))
            .unwrap_or_else(|| filename.to_string());

        let find_key = |key_suffix: &str| -> Option<usize> {
            if let Some(v) = md.get(&format!("{}.{}", arch, key_suffix)) {
                return v.to_u32().ok().map(|v| v as usize);
            }

            for (k, v) in md.iter() {
                if k.ends_with(&format!(".{}", key_suffix)) {
                    if let Ok(val) = v.to_u32() {
                        return Some(val as usize);
                    }
                }
            }
            None
        };

        let get_required = |key_suffix: &str| -> Result<usize> {
            find_key(key_suffix)
                .ok_or_else(|| anyhow::anyhow!("Missing metadata key: {}", key_suffix))
        };

        let get_optional =
            |key_suffix: &str, default: usize| -> usize { find_key(key_suffix).unwrap_or(default) };

        let chat_template = md
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok().map(|s| s.clone()));

        Ok(GgufMetadata {
            name: model_name,
            architecture: arch.clone(),
            n_layer: get_required("block_count")?,
            n_embd: get_required("embedding_length")?,
            vocab_size: get_required("vocab_size")?,
            context_length: get_optional("context_length", 4096),
            file_size,
            chat_template,
        })
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor> {
        let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;

        let logits = match &mut self.inner {
            ModelInner::Llama(m) => m.forward(&input, pos)?,
            ModelInner::Lfm2(m) => m.forward(&input, pos)?,
        };
        Ok(logits)
    }
}
