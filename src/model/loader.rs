use std::fs::File;
use std::io::{Cursor, Seek};
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_lfm2::ModelWeights as Lfm2Model;
use candle_transformers::models::quantized_llama::ModelWeights as LlamaModel;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Model;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3Model;
use memmap2::Mmap;

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
    pub quantization: Option<String>,
}

pub enum ModelInner {
    Llama(LlamaModel),
    Lfm2(Lfm2Model),
    Qwen2(Qwen2Model),
    Qwen3(Qwen3Model),
}

pub struct Model {
    inner: ModelInner,
    metadata: GgufMetadata,
}

pub struct ModelWithMmap {
    pub model: Model,
    pub mmap: Mmap,
}

impl Model {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let (_, model) = Self::load_with_mmap(path)?;
        Ok(model)
    }

    pub fn load_with_mmap(path: &PathBuf) -> Result<(Mmap, Self)> {
        let file_size = std::fs::metadata(path)?.len();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let device = Device::Cpu;

        let file =
            File::open(path).with_context(|| format!("Failed to open model file: {:?}", path))?;

        tracing::info!("Memory-mapping GGUF file ({} MB)...", file_size / 1_000_000);

        let mmap = unsafe { Mmap::map(&file)? };

        // Apply madvise hints BEFORE reading tensor data so the kernel begins
        // async read-ahead while candle's sequential seek+read_exact calls follow.
        // Calling these after from_gguf() would be useless — data already read.
        {
            let ptr = mmap.as_ptr() as *mut std::ffi::c_void;
            let size = mmap.len();
            unsafe {
                libc::madvise(ptr, size, libc::MADV_SEQUENTIAL);
                #[cfg(target_os = "linux")]
                libc::madvise(ptr, size, libc::MADV_HUGEPAGE);
                libc::madvise(ptr, size, libc::MADV_WILLNEED);
            }
            tracing::info!(
                "madvise hints applied ({} MB): SEQUENTIAL + HUGEPAGE + WILLNEED",
                size / 1_000_000
            );
        }

        let mut cursor = Cursor::new(&mmap);

        let content = gguf_file::Content::read(&mut cursor)
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

        cursor.seek(std::io::SeekFrom::Start(0))?;

        let inner = if arch == "lfm2" {
            let weights = Lfm2Model::from_gguf(content, &mut cursor, &device)
                .with_context(|| "Failed to load LFM2 model weights from GGUF")?;
            ModelInner::Lfm2(weights)
        } else if arch == "qwen2" {
            let weights = Qwen2Model::from_gguf(content, &mut cursor, &device)
                .with_context(|| "Failed to load Qwen2 model weights from GGUF")?;
            ModelInner::Qwen2(weights)
        } else if arch == "qwen3" {
            let weights = Qwen3Model::from_gguf(content, &mut cursor, &device)
                .with_context(|| "Failed to load Qwen3 model weights from GGUF")?;
            ModelInner::Qwen3(weights)
        } else if arch == "qwen35" {
            anyhow::bail!(
                "Architecture 'qwen35' (Qwen3.5) is not yet supported. \
                 Qwen3.5 uses a hybrid SSM+Attention architecture not implemented in candle 0.9. \
                 Try a Qwen2.5 or Qwen3 model instead."
            )
        } else {
            let weights = LlamaModel::from_gguf(content, &mut cursor, &device)
                .with_context(|| "Failed to load LLaMA model weights from GGUF")?;
            ModelInner::Llama(weights)
        };

        tracing::info!("Model loaded successfully");

        let model = Self { inner, metadata };
        Ok((mmap, model))
    }

    /// No-op. madvise hints are now applied inside `load_with_mmap()` immediately
    /// after the mmap is created and before tensor data is read, which is the only
    /// point where they have effect. Calling this after load returns is useless.
    #[allow(unused_variables)]
    pub fn prefetch_mmap(_mmap: &Mmap) {}

    fn extract_metadata(
        content: &gguf_file::Content,
        filename: &str,
        file_size: u64,
    ) -> Result<GgufMetadata> {
        let md = &content.metadata;

        let arch: String = match md.get("general.architecture") {
            Some(v) => v
                .to_string()
                .cloned()
                .unwrap_or_else(|_| "llama".to_string()),
            None => "llama".to_string(),
        };

        let model_name: String = md
            .get("general.name")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| filename.to_string());

        let find_key = |key_suffix: &str| -> Option<usize> {
            let as_usize = |v: &gguf_file::Value| v.to_u64().ok().map(|n| n as usize);

            if let Some(v) = md.get(&format!("{}.{}", arch, key_suffix)) {
                return as_usize(v);
            }

            for (k, v) in md.iter() {
                if k.ends_with(&format!(".{}", key_suffix)) {
                    if let Some(val) = as_usize(v) {
                        return Some(val);
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
            .and_then(|v| v.to_string().ok().cloned());

        let quantization: Option<String> = md
            .get("general.quantization")
            .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
            .or_else(|| {
                // Try to find quantization in various model metadata keys
                let quant_keys = [
                    "quantization_version",
                    "quantization",
                    "quantization_format",
                ];
                for key in quant_keys {
                    if let Some(v) = md.get(&format!("{}.{}", arch, key)) {
                        if let Ok(s) = v.to_string() {
                            if !s.is_empty() {
                                return Some(s.to_string());
                            }
                        }
                    }
                }
                None
            })
            .or_else(|| {
                filename
                    .split('.')
                    .filter(|s| !s.eq_ignore_ascii_case("gguf") && !s.eq_ignore_ascii_case("bin"))
                    .last()
                    .map(|s| s.to_string())
                    .filter(|s| {
                        s.len() >= 2
                            && (s.starts_with("q") || s.starts_with("Q"))
                            && s.chars()
                                .skip(1)
                                .all(|c| c.is_ascii_digit() || c == '_' || c == '-')
                    })
            });

        Ok(GgufMetadata {
            name: model_name,
            architecture: arch.clone(),
            n_layer: get_required("block_count")?,
            n_embd: get_required("embedding_length")?,
            vocab_size: find_key("vocab_size")
                .or_else(|| {
                    // Fallback: derive from tokenizer token list length.
                    // Newer GGUF files (e.g. qwen35) omit the explicit vocab_size key.
                    md.get("tokenizer.ggml.tokens")
                        .and_then(|v| v.to_vec().ok())
                        .map(|arr| arr.len())
                })
                .ok_or_else(|| anyhow::anyhow!("Missing metadata key: vocab_size"))?,
            context_length: get_optional("context_length", 4096),
            file_size,
            chat_template,
            quantization,
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
            ModelInner::Qwen2(m) => m.forward(&input, pos)?,
            ModelInner::Qwen3(m) => m.forward(&input, pos)?,
        };
        Ok(logits)
    }
}
