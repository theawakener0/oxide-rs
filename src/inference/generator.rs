use std::path::PathBuf;

use anyhow::Result;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use minijinja::{context, Environment};
use rayon::prelude::*;

use crate::inference::paged_cache::PagedKvCache;
use crate::model::{GgufMetadata, Model, TokenizerWrapper};

pub enum StreamEvent {
    Token(String),
    PrefillStatus(usize),
    Done,
}

#[derive(Clone, serde::Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
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
    kv_cache: Option<PagedKvCache>,
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

        let (mmap, model) = Model::load_with_mmap(model_path)?;
        Model::prefetch_mmap(&mmap);

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

        let token_history = Vec::with_capacity(metadata.context_length);

        let kv_cache = Some(PagedKvCache::new(
            metadata.n_embd / metadata.n_layer,
            metadata.n_embd / metadata.n_layer,
            metadata.context_length,
        ));

        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            template,
            metadata,
            messages: Vec::new(),
            system_prompt,
            token_history,
            kv_cache,
        })
    }

    pub fn kv_cache_stats(&self) -> Option<(usize, usize)> {
        self.kv_cache
            .as_ref()
            .map(|c| (c.current_seq_len(), c.max_seq_len()))
    }

    pub fn clear_kv_cache(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.reset();
        }
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn context_used(&self) -> usize {
        self.token_history.len()
    }

    pub fn context_limit(&self) -> usize {
        self.metadata.context_length
    }

    pub fn context_percentage(&self) -> f32 {
        let limit = self.context_limit();
        if limit == 0 {
            0.0
        } else {
            (self.context_used() as f32 / limit as f32) * 100.0
        }
    }

    pub fn context_warning(&self) -> bool {
        self.context_percentage() >= 80.0
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
        callback: F,
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
                let old_len = self.token_history.len();
                self.token_history.drain(0..excess);
                tracing::debug!(
                    "Context truncated: {} tokens -> {} tokens (kept most recent)",
                    old_len,
                    self.token_history.len()
                );
            } else {
                self.token_history.clear();
            }
        }

        let result = self.generate_internal_with_tokens(
            &prompt_tokens,
            max_tokens,
            repeat_penalty,
            repeat_last_n,
            true,
            callback,
        )?;

        self.messages.push(Message {
            role: "assistant".into(),
            content: result.clone(),
        });

        Ok(result)
    }

    fn generate_internal_with_tokens<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        store_history: bool,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(StreamEvent),
    {
        let history_len = if store_history {
            self.token_history.len()
        } else {
            0
        };

        let total_len = if store_history {
            self.token_history.len() + prompt_tokens.len() + max_tokens
        } else {
            prompt_tokens.len() + max_tokens
        };

        let needs_truncation = total_len > self.metadata.context_length;
        let excess = if needs_truncation {
            Some(total_len - self.metadata.context_length)
        } else {
            None
        };

        let mut all_tokens = Vec::with_capacity(self.metadata.context_length);

        if store_history {
            if let Some(excess) = excess {
                if excess < self.token_history.len() {
                    all_tokens.extend_from_slice(&self.token_history[excess..]);
                }
            } else {
                all_tokens.extend_from_slice(&self.token_history);
            }
        }

        all_tokens.extend_from_slice(prompt_tokens);

        let eos_token = self.tokenizer.eos_token_id();

        let prompt_start = std::time::Instant::now();

        callback(StreamEvent::PrefillStatus(prompt_tokens.len()));

        let logits = self.model.forward(prompt_tokens, 0)?;
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

        let response_tokens: Vec<u32> = all_tokens[history_len + prompt_tokens.len()..].to_vec();

        if store_history {
            self.token_history = all_tokens;
        }

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

        Ok(response)
    }

    pub fn generate_batch(
        &mut self,
        prompts: Vec<&str>,
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Vec<String>> {
        let template = self.template.clone();
        let system_prompt = self.system_prompt.clone();

        let prompt_tokens_list: Vec<Vec<u32>> = prompts
            .par_iter()
            .map(|prompt| {
                let mut all_messages = Vec::new();
                if let Some(ref sys) = system_prompt {
                    all_messages.push(Message {
                        role: "system".into(),
                        content: sys.clone(),
                    });
                }
                all_messages.push(Message {
                    role: "user".into(),
                    content: prompt.to_string(),
                });

                let prompt_text = template.apply(&all_messages)?;
                self.tokenizer.encode(&prompt_text)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut results = Vec::with_capacity(prompts.len());

        for prompt_tokens in prompt_tokens_list {
            let result = self.generate_internal_with_tokens(
                &prompt_tokens,
                max_tokens,
                repeat_penalty,
                repeat_last_n,
                false,
                |_| {},
            )?;
            results.push(result);
        }

        Ok(results)
    }
}

unsafe impl Send for Generator {}
