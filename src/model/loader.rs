use std::path::PathBuf;

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub param_count: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub vocab_size: usize,
    pub file_size: u64,
}

pub struct ModelLoader {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub info: ModelInfo,
}

impl ModelLoader {
    pub fn new(model_path: PathBuf, tokenizer_path: PathBuf) -> Result<Self> {
        let file_size = std::fs::metadata(&model_path)?.len();

        let filename = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let (name, arch) = Self::detect_architecture(filename);

        let info = ModelInfo {
            name,
            architecture: arch,
            param_count: "Unknown".to_string(),
            n_layer: 0,
            n_embd: 0,
            n_head: 0,
            vocab_size: 0,
            file_size,
        };

        Ok(Self {
            model_path,
            tokenizer_path,
            info,
        })
    }

    fn detect_architecture(filename: &str) -> (String, String) {
        let lower = filename.to_lowercase();
        if lower.contains("gemma") {
            ("Gemma".to_string(), "gemma".to_string())
        } else if lower.contains("smollm") || lower.contains("smolvlm") {
            ("SmolLM".to_string(), "smollm".to_string())
        } else if lower.contains("lfm") {
            ("LFM".to_string(), "lfm".to_string())
        } else {
            ("LLaMA".to_string(), "llama".to_string())
        }
    }

    pub fn get_model_info(&self) -> ModelInfo {
        self.info.clone()
    }

    pub fn generate(
        &self,
        prompt: &str,
        _max_tokens: usize,
        _temperature: f32,
        _top_p: f32,
        _top_k: usize,
        _repeat_penalty: f32,
    ) -> Result<String> {
        Ok(format!(
            "[Demo Mode]\n\nModel: {}\nArchitecture: {}\nPrompt: {}...\n\nNote: Full GGUF inference requires additional Candle model implementation.",
            self.info.name,
            self.info.architecture,
            &prompt[..prompt.len().min(50)]
        ))
    }

    pub fn generate_streaming<F>(
        &self,
        prompt: &str,
        _max_tokens: usize,
        _temperature: f32,
        _top_p: f32,
        _top_k: usize,
        _repeat_penalty: f32,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str, bool),
    {
        let response = format!(
            "[Demo Mode]\nModel: {}\nArchitecture: {}\nPrompt: {}...\n\nNote: Full GGUF inference requires additional Candle model implementation.",
            self.info.name,
            self.info.architecture,
            &prompt[..prompt.len().min(50)]
        );

        callback(&response, false);
        callback("", true);

        Ok(response)
    }
}
