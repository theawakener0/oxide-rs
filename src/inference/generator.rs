use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;

use crate::model::{GgufMetadata, Model, TokenizerWrapper};

pub enum StreamEvent {
    Token(String),
    Done { tokens_generated: usize },
    Error(String),
}

pub struct ChatTemplate {
    pub name: String,
    pub user_prefix: String,
    pub user_suffix: String,
    pub assistant_prefix: String,
    pub assistant_suffix: String,
    pub system_prefix: Option<String>,
}

impl ChatTemplate {
    pub fn from_architecture(arch: &str, model_name: &str) -> Self {
        let lower = model_name.to_lowercase();

        if lower.contains("smollm") {
            Self {
                name: "smollm".to_string(),
                user_prefix: "<|im_start|>user\n".to_string(),
                user_suffix: "<|im_end|>\n".to_string(),
                assistant_prefix: "<|im_start|>assistant\n".to_string(),
                assistant_suffix: "<|im_end|>\n".to_string(),
                system_prefix: Some(
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n".to_string(),
                ),
            }
        } else if lower.contains("lfm") {
            Self {
                name: "lfm".to_string(),
                user_prefix: "<|user|>\n".to_string(),
                user_suffix: "<|end|>\n".to_string(),
                assistant_prefix: "<|assistant|\n".to_string(),
                assistant_suffix: "<|end|>\n".to_string(),
                system_prefix: None,
            }
        } else if lower.contains("gemma") {
            Self {
                name: "gemma".to_string(),
                user_prefix: "<start_of_turn>user\n".to_string(),
                user_suffix: "<end_of_turn>\n".to_string(),
                assistant_prefix: "<start_of_turn>model\n".to_string(),
                assistant_suffix: "<end_of_turn>\n".to_string(),
                system_prefix: None,
            }
        } else if lower.contains("mistral") || lower.contains("zephyr") {
            Self {
                name: "mistral".to_string(),
                user_prefix: "[INST] ".to_string(),
                user_suffix: " [/INST]".to_string(),
                assistant_prefix: "".to_string(),
                assistant_suffix: "</s>".to_string(),
                system_prefix: None,
            }
        } else {
            Self {
                name: "llama".to_string(),
                user_prefix: "<|user|>\n".to_string(),
                user_suffix: "\n".to_string(),
                assistant_prefix: "<|assistant|\n".to_string(),
                assistant_suffix: "\n".to_string(),
                system_prefix: None,
            }
        }
    }

    pub fn apply(&self, prompt: &str, include_response_start: bool) -> String {
        let mut result = String::new();

        if let Some(sys) = &self.system_prefix {
            result.push_str(sys);
        }

        result.push_str(&self.user_prefix);
        result.push_str(prompt);
        result.push_str(&self.user_suffix);

        if include_response_start {
            result.push_str(&self.assistant_prefix);
        }

        result
    }
}

pub struct Generator {
    model: Model,
    tokenizer: TokenizerWrapper,
    logits_processor: LogitsProcessor,
    history: Vec<u32>,
    max_history: usize,
    template: ChatTemplate,
    metadata: GgufMetadata,
}

impl Generator {
    pub fn new(
        model_path: &PathBuf,
        tokenizer_path: Option<&PathBuf>,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        seed: u64,
        max_history: usize,
    ) -> Result<Self> {
        tracing::info!("Loading model from: {:?}", model_path);

        let model = Model::load(model_path)?;
        let metadata = model.metadata().clone();
        let template = ChatTemplate::from_architecture(&metadata.architecture, &metadata.name);

        let tokenizer = if let Some(path) = tokenizer_path {
            TokenizerWrapper::from_file(path)?
        } else {
            tracing::info!("Extracting tokenizer from GGUF...");
            let mut file = std::fs::File::open(model_path)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
            TokenizerWrapper::from_gguf(&content)?
        };

        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };

        let logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            history: Vec::new(),
            max_history,
            template,
            metadata,
        })
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn generate<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(StreamEvent),
    {
        let prompt_text = self.template.apply(prompt, true);
        let prompt_tokens = self.tokenizer.encode(&prompt_text)?;

        let total_len = self.history.len() + prompt_tokens.len() + max_tokens;
        if total_len > self.metadata.context_length {
            let excess = total_len - self.metadata.context_length;
            if excess < self.history.len() {
                self.history.drain(0..excess);
            } else {
                self.history.clear();
            }
        }

        let mut all_tokens = Vec::new();
        all_tokens.extend_from_slice(&self.history);
        all_tokens.extend_from_slice(&prompt_tokens);

        let eos_token = self.tokenizer.eos_token_id();

        let prompt_start = std::time::Instant::now();

        let input =
            Tensor::new(prompt_tokens.as_slice(), &candle_core::Device::Cpu)?.unsqueeze(0)?;
        let logits = self.model.forward(&prompt_tokens, 0)?;
        let logits = logits.squeeze(0)?;

        let mut next_token = self.logits_processor.sample(&logits)?;

        tracing::debug!(
            "Prompt processed: {} tokens in {:.2}s",
            prompt_tokens.len(),
            prompt_start.elapsed().as_secs_f32()
        );

        all_tokens.push(next_token);

        if let Some(text) = self.tokenizer.decode_next(next_token)? {
            callback(StreamEvent::Token(text));
        }

        let gen_start = std::time::Instant::now();
        let mut generated = 1usize;

        for _ in 1..max_tokens {
            if next_token == eos_token {
                break;
            }

            let input = Tensor::new(&[next_token], &candle_core::Device::Cpu)?.unsqueeze(0)?;

            let logits = self.model.forward(&[next_token], all_tokens.len() - 1)?;
            let logits = logits.squeeze(0)?;

            let logits = if repeat_penalty != 1.0 {
                let start_at = all_tokens.len().saturating_sub(repeat_last_n);
                apply_repeat_penalty(&logits, repeat_penalty, &all_tokens[start_at..])?
            } else {
                logits
            };

            next_token = self.logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            generated += 1;

            if let Some(text) = self.tokenizer.decode_next(next_token)? {
                callback(StreamEvent::Token(text));
            }

            if next_token == eos_token {
                break;
            }
        }

        if let Some(rest) = self.tokenizer.decode_rest()? {
            callback(StreamEvent::Token(rest));
        }

        self.tokenizer.clear_cache();

        let response_tokens: Vec<u32> =
            all_tokens[self.history.len() + prompt_tokens.len()..].to_vec();
        self.history = all_tokens;

        let dt = gen_start.elapsed();
        let tokens_per_sec = generated as f64 / dt.as_secs_f64();
        tracing::info!(
            "Generated {} tokens in {:.2}s ({:.1} tokens/s)",
            generated,
            dt.as_secs_f32(),
            tokens_per_sec
        );

        callback(StreamEvent::Done {
            tokens_generated: generated,
        });

        let response = self.tokenizer.decode(&response_tokens)?;
        Ok(response)
    }
}

unsafe impl Send for Generator {}
