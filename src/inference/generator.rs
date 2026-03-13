use std::path::PathBuf;

use anyhow::Result;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use minijinja::{context, Environment};

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

/// Holds a pre-compiled minijinja environment so the template string is parsed
/// and compiled exactly once at construction time, not on every generation call.
pub struct ChatTemplate {
    /// `None` when no chat template was embedded in the GGUF file.
    env: Option<Environment<'static>>,
}

const STRIP_SEQUENCES: &[&str] = &[
    "<|im_start|>assistant",
    "<|im_start|>system",
    "<|im_start|>user",
    "</|im_start|>",
    "</|im_str>",
    "<|im_str|>",
    "<|im_start|>",
    "<|start|>",
    "<|sep|>",
    "<s>",
    "<pad>",
];

const STOP_SEQUENCES: &[&str] = &["<|im_end|>", "<|end|>", "</s>"];

#[derive(Debug, Default)]
struct ResponseProcessor {
    buffer: String,
}

#[derive(Debug, Default)]
struct ProcessedChunk {
    text: String,
    should_stop: bool,
}

impl ResponseProcessor {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    fn push(&mut self, chunk: &str) -> ProcessedChunk {
        self.buffer.push_str(chunk);
        strip_full_sequences(&mut self.buffer);

        if let Some(stop_idx) = earliest_sequence_index(&self.buffer, STOP_SEQUENCES) {
            let text = strip_inline_sequences(&self.buffer[..stop_idx]);
            self.buffer.clear();
            return ProcessedChunk {
                text,
                should_stop: true,
            };
        }

        let keep_len = trailing_partial_match_len(&self.buffer);
        let safe_len = self.buffer.len().saturating_sub(keep_len);
        let safe_len = floor_char_boundary(&self.buffer, safe_len);

        if safe_len == 0 {
            return ProcessedChunk::default();
        }

        let text = strip_inline_sequences(&self.buffer[..safe_len]);
        self.buffer.drain(..safe_len);

        ProcessedChunk {
            text,
            should_stop: false,
        }
    }

    fn finish(&mut self) -> String {
        strip_full_sequences(&mut self.buffer);
        let text = strip_inline_sequences(&self.buffer);
        self.buffer.clear();
        text
    }
}

fn trailing_partial_match_len(input: &str) -> usize {
    STRIP_SEQUENCES
        .iter()
        .chain(STOP_SEQUENCES.iter())
        .map(|pattern| {
            let max_len = input.len().min(pattern.len().saturating_sub(1));
            (1..=max_len)
                .rev()
                .find(|&len| input.ends_with(&pattern[..len]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

fn earliest_sequence_index(input: &str, patterns: &[&str]) -> Option<usize> {
    patterns.iter().filter_map(|p| input.find(p)).min()
}

fn strip_inline_sequences(input: &str) -> String {
    let mut result = input.to_string();
    for pattern in STRIP_SEQUENCES {
        result = result.replace(pattern, "");
    }
    result
}

fn strip_full_sequences(buffer: &mut String) {
    loop {
        let mut next = buffer.clone();
        for pattern in STRIP_SEQUENCES {
            next = next.replace(pattern, "");
        }
        if next == *buffer {
            break;
        }
        *buffer = next;
    }
}

fn floor_char_boundary(input: &str, mut idx: usize) -> usize {
    idx = idx.min(input.len());
    while idx > 0 && !input.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn normalize_chat_template(src: &str) -> String {
    src.replace(
        "content.startswith('<tool_response>') and content.endswith('</tool_response>')",
        "(content[:15] == '<tool_response>') and (content[-16:] == '</tool_response>')",
    )
    .replace(
        "content.starts_with('<tool_response>') and content.ends_with('</tool_response>')",
        "(content[:15] == '<tool_response>') and (content[-16:] == '</tool_response>')",
    )
}

impl ChatTemplate {
    pub fn new(template: Option<String>) -> Result<Self> {
        let env = match template {
            None => None,
            Some(src) => {
                let src = normalize_chat_template(&src);
                let mut e = Environment::new();
                e.add_template_owned("chat".to_string(), src)?;
                Some(e)
            }
        };
        Ok(Self { env })
    }

    pub fn apply(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let env = match &self.env {
            Some(e) => e,
            None => {
                anyhow::bail!("GGUF file has no chat_template. Use a model with embedded template.")
            }
        };

        let tmpl = env.get_template("chat")?;
        let rendered = tmpl.render(context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => false,
            add_vision_id => false,
        })?;
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
    /// Reusable token buffer for the current generation call. Allocated once
    /// with context_length capacity and cleared (not freed) between calls.
    all_tokens: Vec<u32>,
    kv_cache: Option<PagedKvCache>,
    batch_size: usize,
}

impl Generator {
    fn encode_chat_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    fn conversation_messages(&self) -> Vec<Message> {
        let mut messages =
            Vec::with_capacity(self.messages.len() + usize::from(self.system_prompt.is_some()));
        if let Some(ref sys) = self.system_prompt {
            messages.push(Message {
                role: "system".into(),
                content: sys.clone(),
            });
        }
        messages.extend(self.messages.iter().cloned());
        messages
    }

    fn rebuild_token_history(&mut self) -> Result<()> {
        let messages = self.conversation_messages();
        if messages.is_empty() {
            self.token_history.clear();
            return Ok(());
        }

        let rendered = self.template.apply(&messages, false)?;
        self.token_history = self.encode_chat_text(&rendered)?;
        Ok(())
    }

    fn drop_oldest_turn(&mut self) -> bool {
        if self.messages.is_empty() {
            return false;
        }

        self.messages.remove(0);
        if matches!(self.messages.first(), Some(message) if message.role == "assistant") {
            self.messages.remove(0);
        }
        true
    }

    pub fn new(
        model_path: &PathBuf,
        tokenizer_path: Option<&PathBuf>,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        seed: u64,
        system_prompt: Option<String>,
        batch_size: usize,
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
        let all_tokens = Vec::with_capacity(metadata.context_length);

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
            all_tokens,
            kv_cache,
            batch_size,
        })
    }

    pub fn kv_cache_stats(&self) -> Option<(usize, usize)> {
        self.kv_cache
            .as_ref()
            .map(|c| (c.current_seq_len(), c.max_seq_len()))
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
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
        self.clear_kv_cache();
    }

    pub fn warmup(&mut self, num_warmup_tokens: usize) -> Result<()> {
        tracing::info!("Warming up model with {} tokens...", num_warmup_tokens);

        let warmup_tokens = vec![0u32; num_warmup_tokens.min(512)];

        let batch_size = self.batch_size;
        for i in (0..warmup_tokens.len()).step_by(batch_size) {
            let end = (i + batch_size).min(warmup_tokens.len());
            let batch = &warmup_tokens[i..end];
            let _ = self.model.forward(batch, i)?;
        }

        tracing::info!("Model warmup complete");
        Ok(())
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Appends the user message to history, builds the full chat prompt, encodes
    /// it, and trims the token history if needed to fit within the context window.
    /// Returns the encoded prompt tokens ready for generation.
    fn prepare_prompt(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
        self.messages.push(Message {
            role: "user".into(),
            content: prompt.into(),
        });

        loop {
            let owned = self.conversation_messages();
            let prompt_text = self.template.apply(&owned, true)?;
            let prompt_tokens = self.encode_chat_text(&prompt_text)?;

            let total_len = prompt_tokens.len() + max_tokens;
            if total_len <= self.metadata.context_length {
                return Ok(prompt_tokens);
            }

            if !self.drop_oldest_turn() {
                anyhow::bail!(
                    "Prompt is too large for the model context window ({} > {}).",
                    total_len,
                    self.metadata.context_length
                );
            }
        }
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
        let prompt_tokens = self.prepare_prompt(prompt, max_tokens)?;

        let result = self.generate_internal_with_tokens(
            &prompt_tokens,
            max_tokens,
            repeat_penalty,
            repeat_last_n,
            callback,
            false,
        )?;

        self.messages.push(Message {
            role: "assistant".into(),
            content: result.clone(),
        });
        self.rebuild_token_history()?;

        Ok(result)
    }

    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(StreamEvent),
    {
        let prompt_tokens = self.prepare_prompt(prompt, max_tokens)?;

        let result = self.generate_internal_with_tokens(
            &prompt_tokens,
            max_tokens,
            repeat_penalty,
            repeat_last_n,
            callback,
            true,
        )?;

        self.messages.push(Message {
            role: "assistant".into(),
            content: result,
        });
        self.rebuild_token_history()?;

        Ok(())
    }

    fn generate_internal_with_tokens<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        mut callback: F,
        _streaming: bool,
    ) -> Result<String>
    where
        F: FnMut(StreamEvent),
    {
        let total_len = prompt_tokens.len() + max_tokens;
        if total_len > self.metadata.context_length {
            anyhow::bail!(
                "Prompt is too large for the model context window ({} > {}).",
                total_len,
                self.metadata.context_length
            );
        }

        // Reuse the pre-allocated buffer instead of allocating context_length
        // capacity (up to 128KB) on every call.
        self.all_tokens.clear();
        self.all_tokens.extend_from_slice(prompt_tokens);

        let eos_token = self.tokenizer.eos_token_id();
        let mut response_processor = ResponseProcessor::new();
        let mut response_text = String::new();

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

        let mut generated = 1usize;
        self.all_tokens.push(next_token);

        // Emit first generated token via incremental decoder.
        if !self.tokenizer.is_special_token(next_token) {
            if let Some(text) = self.tokenizer.decode_next(next_token)? {
                let processed = response_processor.push(&text);
                if !processed.text.is_empty() {
                    response_text.push_str(&processed.text);
                    callback(StreamEvent::Token(processed.text));
                }
                if processed.should_stop {
                    self.tokenizer.clear_cache();
                    callback(StreamEvent::Done);

                    return Ok(response_text);
                }
            }
        }

        let gen_start = std::time::Instant::now();

        for _ in 1..max_tokens {
            if next_token == eos_token {
                break;
            }

            let logits = self
                .model
                .forward(&[next_token], self.all_tokens.len() - 1)?;
            let logits = logits.squeeze(0)?;

            let logits = if repeat_penalty != 1.0 {
                let start_at = self.all_tokens.len().saturating_sub(repeat_last_n);
                apply_repeat_penalty(&logits, repeat_penalty, &self.all_tokens[start_at..])?
            } else {
                logits
            };

            next_token = self.logits_processor.sample(&logits)?;
            self.all_tokens.push(next_token);
            generated += 1;

            // Use incremental decode: emits text as soon as a word boundary is
            // reached, without buffering or re-decoding previously seen tokens.
            if !self.tokenizer.is_special_token(next_token) {
                if let Some(text) = self.tokenizer.decode_next(next_token)? {
                    let processed = response_processor.push(&text);
                    if !processed.text.is_empty() {
                        response_text.push_str(&processed.text);
                        callback(StreamEvent::Token(processed.text));
                    }
                    if processed.should_stop {
                        break;
                    }
                }
            }
        }

        // clear_cache() resets the incremental decoder state. decode_rest() is
        // intentionally NOT called here: it returns the full cached_decoded
        // accumulation (all text already streamed via decode_next), which would
        // produce duplicate output. Clearing is sufficient — shimmytok's
        // decode_single emits each fragment as soon as it has enough bytes.
        self.tokenizer.clear_cache();

        let tail = response_processor.finish();
        if !tail.is_empty() {
            response_text.push_str(&tail);
            callback(StreamEvent::Token(tail));
        }

        let dt = gen_start.elapsed();
        let tokens_per_sec = if generated > 0 && dt.as_secs_f64() > 0.0 {
            generated as f64 / dt.as_secs_f64()
        } else {
            0.0
        };
        tracing::info!(
            "Generated {} tokens in {:.2}s ({:.1} tokens/s)",
            generated,
            dt.as_secs_f32(),
            tokens_per_sec
        );

        callback(StreamEvent::Done);

        Ok(response_text)
    }

    pub fn generate_batch(
        &mut self,
        prompts: Vec<String>,
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Vec<String>> {
        if prompts.is_empty() {
            return Ok(vec![]);
        }

        // Build prompt texts first (borrows self.template + self.system_prompt),
        // then encode in a separate pass (borrows self.tokenizer).
        // This avoids needing to clone ChatTemplate, which no longer implements Clone.
        let prompt_texts: Vec<String> = prompts
            .iter()
            .map(|prompt| {
                let mut all_messages = Vec::new();
                if let Some(ref sys) = self.system_prompt {
                    all_messages.push(Message {
                        role: "system".into(),
                        content: sys.clone(),
                    });
                }
                all_messages.push(Message {
                    role: "user".into(),
                    content: prompt.clone(),
                });
                self.template.apply(&all_messages, true)
            })
            .collect::<Result<Vec<_>>>()?;

        let prompt_tokens_list: Vec<Vec<u32>> = prompt_texts
            .iter()
            .map(|text| self.encode_chat_text(text))
            .collect::<Result<Vec<_>>>()?;

        let mut results = Vec::with_capacity(prompts.len());

        for prompt_tokens in prompt_tokens_list {
            let result = self.generate_internal_with_tokens(
                &prompt_tokens,
                max_tokens,
                repeat_penalty,
                repeat_last_n,
                |_| {},
                false,
            )?;
            results.push(result);
        }

        Ok(results)
    }
}

unsafe impl Send for Generator {}

#[cfg(test)]
mod tests {
    use super::ResponseProcessor;

    #[test]
    fn strips_split_control_sequences_across_chunks() {
        let mut processor = ResponseProcessor::new();

        let first = processor.push("Hello<|im_");
        assert_eq!(first.text, "Hello");
        assert!(!first.should_stop);

        let second = processor.push("start|>assistant");
        assert_eq!(second.text, "");
        assert!(!second.should_stop);

        let third = processor.push(" world");
        assert_eq!(third.text, " world");
        assert!(!third.should_stop);
    }

    #[test]
    fn stops_on_im_end_without_emitting_marker() {
        let mut processor = ResponseProcessor::new();

        let chunk = processor.push("Answer<|im_end|>garbage");
        assert_eq!(chunk.text, "Answer");
        assert!(chunk.should_stop);
    }

    #[test]
    fn flushes_remaining_text_on_finish() {
        let mut processor = ResponseProcessor::new();

        let chunk = processor.push("done");
        assert_eq!(chunk.text, "done");
        assert!(!chunk.should_stop);

        assert_eq!(processor.finish(), "");
    }
}
