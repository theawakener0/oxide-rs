use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use tokenizers::Tokenizer;

pub struct TokenizerWrapper {
    inner: Tokenizer,
    eos_token_id: u32,
    bos_token_id: u32,
    unk_token_id: u32,
    vocab: HashMap<String, u32>,
    pending_tokens: Vec<u32>,
}

impl TokenizerWrapper {
    pub fn from_gguf(content: &gguf_file::Content) -> Result<Self> {
        let md = &content.metadata;

        let tokens_val = match md.get("tokenizer.ggml.tokens") {
            Some(v) => v,
            None => bail!("No tokenizer tokens found in GGUF"),
        };

        let tokens = tokens_val
            .to_vec()
            .with_context(|| "Failed to parse tokenizer tokens")?;

        let token_strings: Vec<String> = tokens
            .iter()
            .filter_map(|t| t.to_string().ok())
            .map(|s| s.replace("<0x0A>", "\n").replace('▁', " "))
            .collect();

        let vocab: HashMap<String, u32> = token_strings
            .iter()
            .enumerate()
            .map(|(i, s): (usize, &String)| (s.clone(), i as u32))
            .collect();

        let unk_token_id = Self::find_unk_token(&vocab, &token_strings);
        let eos_token_id = Self::find_eos_token(&vocab, &token_strings);
        let bos_token_id = Self::find_bos_token(&vocab, &token_strings);

        let tokenizer_json = Self::build_tokenizer_json(&token_strings, unk_token_id)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_json)
            .map_err(|e| anyhow::anyhow!("Failed to create tokenizer: {}", e))?;

        tracing::info!(
            "Loaded tokenizer from GGUF: {} tokens, UNK={}, EOS={}, BOS={}",
            token_strings.len(),
            unk_token_id,
            eos_token_id,
            bos_token_id
        );

        Ok(Self {
            inner: tokenizer,
            eos_token_id,
            bos_token_id,
            unk_token_id,
            vocab,
            pending_tokens: Vec::new(),
        })
    }

    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let vocab: HashMap<String, u32> = tokenizer
            .get_vocab(true)
            .into_iter()
            .map(|(k, v)| (k, v))
            .collect();

        let vocab_size = tokenizer.get_vocab_size(true);
        let unk_token_id = Self::find_unk_token(&vocab, &[]);
        let eos_token_id = Self::find_eos_token(&vocab, &[]);
        let bos_token_id = Self::find_bos_token(&vocab, &[]);

        tracing::info!(
            "Loaded tokenizer from file: {} tokens, UNK={}, EOS={}, BOS={}",
            vocab_size,
            unk_token_id,
            eos_token_id,
            bos_token_id
        );

        Ok(Self {
            inner: tokenizer,
            eos_token_id,
            bos_token_id,
            unk_token_id,
            vocab,
            pending_tokens: Vec::new(),
        })
    }

    pub fn from_hf(repo: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo.to_string());
        let tokenizer_path = repo.get("tokenizer.json")?;
        Self::from_file(&tokenizer_path)
    }

    fn build_tokenizer_json(tokens: &[String], unk_id: u32) -> Result<Vec<u8>> {
        let vocab_entries: Vec<serde_json::Value> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| serde_json::json!([t, -((i as f64) / (tokens.len() as f64))]))
            .collect();

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "Unigram",
                "vocab": vocab_entries,
                "unk_id": unk_id
            },
            "decoder": {
                "type": "Sequence",
                "decoders": [
                    {"type": "Replace", "pattern": {"String": " "}, "content": ""},
                    {"type": "ByteFallback"}
                ]
            },
            "added_tokens": [
                {"id": unk_id, "content": "<unk>", "special": true}
            ]
        });

        serde_json::to_vec(&tokenizer_json).with_context(|| "Failed to serialize tokenizer JSON")
    }

    fn find_unk_token(vocab: &HashMap<String, u32>, tokens: &[String]) -> u32 {
        for unk_str in &["<unk>", "<|unk|>", "[UNK]", "[PAD]"] {
            if let Some(id) = vocab.get(*unk_str) {
                return *id;
            }
        }

        for (i, t) in tokens.iter().enumerate() {
            if t == "<unk>" || t.contains("unk") {
                return i as u32;
            }
        }

        0
    }

    fn find_eos_token(vocab: &HashMap<String, u32>, tokens: &[String]) -> u32 {
        for eos_str in &[
            "</s>",
            "\n",
            "<eos>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<end_of_turn>",
            "<｜end▁of▁sentence｜>",
            "\n",
        ] {
            if let Some(id) = vocab.get(*eos_str) {
                return *id;
            }
        }

        for (i, t) in tokens.iter().enumerate() {
            if t.contains("endoftext") || t.contains("eos") || t == "\n" {
                return i as u32;
            }
        }

        1
    }

    fn find_bos_token(vocab: &HashMap<String, u32>, _tokens: &[String]) -> u32 {
        for bos_str in &[
            "<s>",
            "<|begin_of_text|>",
            "<bos>",
            "<|im_start|>",
            "<start_of_turn>",
        ] {
            if let Some(id) = vocab.get(*bos_str) {
                return *id;
            }
        }
        0
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let text = self
            .inner
            .decode(tokens, false)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
        Ok(text)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn clear_cache(&mut self) {
        self.pending_tokens.clear();
    }

    pub fn decode_next(&mut self, token: u32) -> Result<Option<String>> {
        self.pending_tokens.push(token);

        let decoded = self.decode(&self.pending_tokens)?;

        let prev_decoded: String = {
            let prev_tokens = &self.pending_tokens[..self.pending_tokens.len() - 1];
            if prev_tokens.is_empty() {
                String::new()
            } else {
                self.decode(prev_tokens)?
            }
        };

        if decoded.len() > prev_decoded.len() {
            let new_text = decoded[prev_decoded.len()..].to_string();
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&mut self) -> Result<Option<String>> {
        if self.pending_tokens.is_empty() {
            return Ok(None);
        }

        let decoded = self.decode(&self.pending_tokens)?;

        let prev_decoded: String = {
            if self.pending_tokens.len() > 1 {
                let prev_tokens = &self.pending_tokens[..self.pending_tokens.len() - 1];
                self.decode(prev_tokens)?
            } else {
                String::new()
            }
        };

        if decoded.len() > prev_decoded.len() {
            Ok(Some(decoded[prev_decoded.len()..].to_string()))
        } else {
            Ok(None)
        }
    }
}
