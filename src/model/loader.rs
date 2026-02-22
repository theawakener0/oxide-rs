use std::fs::File;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub name: String,
    pub architecture: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub file_size: u64,
}

pub struct Model {
    weights: ModelWeights,
    metadata: GgufMetadata,
    device: Device,
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

        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .with_context(|| "Failed to load model weights from GGUF")?;

        tracing::info!(
            "Loaded model: {} ({} layers, {} embedding dim, {} vocab)",
            metadata.name,
            metadata.n_layer,
            metadata.n_embd,
            metadata.vocab_size
        );

        Ok(Self {
            weights,
            metadata,
            device,
        })
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

        let get_with_prefix = |key: &str| -> Result<usize> {
            let full_key = format!("{}.{}", arch, key);
            if let Some(v) = md.get(&full_key) {
                return v
                    .to_u32()
                    .map(|v| v as usize)
                    .with_context(|| format!("Failed to parse {} as u32", full_key));
            }

            for prefix in &["llama", "qwen", "mistral", "phi", "gemma"] {
                let alt_key = format!("{}.{}", prefix, key);
                if let Some(v) = md.get(&alt_key) {
                    return v
                        .to_u32()
                        .map(|v| v as usize)
                        .with_context(|| format!("Failed to parse {} as u32", alt_key));
                }
            }

            bail!("Missing metadata key: {} (tried {}.{})", key, arch, key)
        };

        let get_with_prefix_or = |key: &str, default: usize| -> usize {
            let full_key = format!("{}.{}", arch, key);
            if let Some(v) = md.get(&full_key).and_then(|v| v.to_u32().ok()) {
                return v as usize;
            }

            for prefix in &["llama", "qwen", "mistral", "phi", "gemma"] {
                let alt_key = format!("{}.{}", prefix, key);
                if let Some(v) = md.get(&alt_key).and_then(|v| v.to_u32().ok()) {
                    return v as usize;
                }
            }

            default
        };

        let (name, _) = Self::detect_architecture(filename);

        let n_head = get_with_prefix("attention.head_count")?;
        let n_head_kv = get_with_prefix_or("attention.head_count_kv", n_head);

        Ok(GgufMetadata {
            name,
            architecture: arch.clone(),
            n_layer: get_with_prefix("block_count")?,
            n_embd: get_with_prefix("embedding_length")?,
            n_head,
            n_head_kv,
            vocab_size: get_with_prefix("vocab_size")?,
            context_length: get_with_prefix_or("context_length", 4096),
            file_size,
        })
    }

    fn detect_architecture(filename: &str) -> (String, String) {
        let lower = filename.to_lowercase();
        if lower.contains("gemma") {
            ("Gemma".to_string(), "gemma".to_string())
        } else if lower.contains("smollm") {
            ("SmolLM".to_string(), "llama".to_string())
        } else if lower.contains("lfm") {
            ("LFM".to_string(), "llama".to_string())
        } else if lower.contains("phi") {
            ("Phi".to_string(), "phi".to_string())
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            ("Mistral".to_string(), "mistral".to_string())
        } else {
            ("LLaMA".to_string(), "llama".to_string())
        }
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor> {
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;

        let logits = self.weights.forward(&input, pos)?;
        Ok(logits)
    }
}
