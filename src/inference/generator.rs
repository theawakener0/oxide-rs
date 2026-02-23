use std::path::PathBuf;

use anyhow::Result;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use minijinja::{context, Environment};

use crate::model::{GgufMetadata, Model, TokenizerWrapper};

pub enum StreamEvent {
    Token(String),
    Done,
}

#[derive(Clone, serde::Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    template_str: Option<String>,
}

impl ChatTemplate {
    pub fn new(template: Option<String>) -> Result<Self> {
        Ok(Self {
            template_str: template,
        })
    }

    pub fn apply(&self, messages: &[Message]) -> Result<String> {
        let template = match &self.template_str {
            Some(t) => t,
            None => {
                anyhow::bail!("GGUF file has no chat_template. Use a model with embedded template.")
            }
        };

        let mut env = Environment::new();
        env.add_template("chat", template)?;
        let tmpl = env.get_template("chat")?;
        let rendered = tmpl.render(context! { messages => messages })?;
        Ok(rendered)
    }
}

pub struct Generator {
    model: Model,
    tokenizer: TokenizerWrapper,
    logits_processor: LogitsProcessor,
    template: ChatTemplate,
    metadata: GgufMetadata,
    messages: Vec<Message>,
    system_prompt: Option<String>,
    token_history: Vec<u32>,
}

impl Generator {
    pub fn new(
        model_path: &PathBuf,
        tokenizer_path: Option<&PathBuf>,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        seed: u64,
        system_prompt: Option<String>,
    ) -> Result<Self> {
        tracing::info!("Loading model from: {:?}", model_path);

        let model = Model::load(model_path)?;
        let metadata = model.metadata().clone();
        let template = ChatTemplate::new(metadata.chat_template.clone())?;

        let tokenizer = if let Some(path) = tokenizer_path {
            TokenizerWrapper::from_file(path)?
        } else {
            tracing::info!("Loading tokenizer from GGUF...");
            TokenizerWrapper::from_gguf(model_path)?
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
            template,
            metadata,
            messages: Vec::new(),
            system_prompt,
            token_history: Vec::new(),
        })
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn clear_history(&mut self) {
        self.messages.clear();
        self.token_history.clear();
    }

    pub fn warmup(&mut self, num_warmup_tokens: usize) -> Result<()> {
        tracing::info!("Warming up model with {} tokens...", num_warmup_tokens);

        let warmup_tokens = vec![0u32; num_warmup_tokens.min(512)];

        for i in (0..warmup_tokens.len()).step_by(64) {
            let end = (i + 64).min(warmup_tokens.len());
            let batch = &warmup_tokens[i..end];
            let _ = self.model.forward(batch, i)?;
        }

        tracing::info!("Model warmup complete");
        Ok(())
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
        self.messages.push(Message {
            role: "user".into(),
            content: prompt.into(),
        });

        let mut all_messages = Vec::new();
        if let Some(ref sys) = self.system_prompt {
            all_messages.push(Message {
                role: "system".into(),
                content: sys.clone(),
            });
        }
        all_messages.extend(self.messages.clone());

        let prompt_text = self.template.apply(&all_messages)?;
        let prompt_tokens = self.tokenizer.encode(&prompt_text)?;

        let total_len = self.token_history.len() + prompt_tokens.len() + max_tokens;
        if total_len > self.metadata.context_length {
            let excess = total_len - self.metadata.context_length;
            if excess < self.token_history.len() {
                self.token_history.drain(0..excess);
            } else {
                self.token_history.clear();
            }
        }

        let mut all_tokens = Vec::new();
        all_tokens.extend_from_slice(&self.token_history);
        all_tokens.extend_from_slice(&prompt_tokens);

        let eos_token = self.tokenizer.eos_token_id();

        let prompt_start = std::time::Instant::now();

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
            all_tokens[self.token_history.len() + prompt_tokens.len()..].to_vec();
        self.token_history = all_tokens;

        let dt = gen_start.elapsed();
        let tokens_per_sec = generated as f64 / dt.as_secs_f64();
        tracing::info!(
            "Generated {} tokens in {:.2}s ({:.1} tokens/s)",
            generated,
            dt.as_secs_f32(),
            tokens_per_sec
        );

        callback(StreamEvent::Done);

        let response = self.tokenizer.decode(&response_tokens)?;

        self.messages.push(Message {
            role: "assistant".into(),
            content: response.clone(),
        });

        Ok(response)
    }
}

unsafe impl Send for Generator {}
