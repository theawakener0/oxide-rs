use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;

use anyhow::Result;
use sha2::{Digest, Sha256};
use shimmytok::Tokenizer as ShimmyTokenizer;

const CACHE_DIR: &str = ".cache/oxide";

pub struct TokenizerWrapper {
    inner: ShimmyTokenizer,
    eos_token_id: u32,
    pending_tokens: Vec<u32>,
}

fn get_cache_path(model_path: &PathBuf) -> Result<PathBuf> {
    let mut hasher = Sha256::new();

    let mut file = File::open(model_path)?;
    let mut header = vec![0u8; 65536];
    file.read(&mut header)?;
    hasher.update(&header);

    let file_size = file.metadata()?.len();
    hasher.update(file_size.to_le_bytes());

    let hash = format!("{:x}", hasher.finalize());

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let cache_dir = PathBuf::from(home).join(CACHE_DIR);
    fs::create_dir_all(&cache_dir)?;

    Ok(cache_dir.join(format!("{}.tokenizer", hash)))
}

impl TokenizerWrapper {
    pub fn from_gguf(path: &PathBuf) -> Result<Self> {
        let cache_path = get_cache_path(path)?;

        let inner = if cache_path.exists() {
            tracing::info!("Loading tokenizer from cache: {:?}", cache_path);
            ShimmyTokenizer::from_gguf_file(&cache_path)
                .map_err(|e| {
                    tracing::warn!("Cache corrupted, reloading from GGUF: {}", e);
                    e
                })
                .or_else(|_| ShimmyTokenizer::from_gguf_file(path))?
        } else {
            tracing::info!("Loading tokenizer from GGUF (first time)...");
            let tokenizer = ShimmyTokenizer::from_gguf_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

            if let Err(e) = fs::copy(path, &cache_path) {
                tracing::warn!("Failed to cache tokenizer: {}", e);
            } else {
                tracing::info!("Tokenizer cached to {:?}", cache_path);
            }

            tokenizer
        };

        let eos_token_id = inner.eos_token();
        tracing::info!("Loaded tokenizer, EOS={}", eos_token_id);

        Ok(Self {
            inner,
            eos_token_id,
            pending_tokens: Vec::new(),
        })
    }

    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let inner = ShimmyTokenizer::from_gguf_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let eos_token_id = inner.eos_token();

        tracing::info!("Loaded tokenizer from file, EOS={}", eos_token_id);

        Ok(Self {
            inner,
            eos_token_id,
            pending_tokens: Vec::new(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encode failed: {}", e))
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, false)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
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

        Ok(get_new_text(&decoded, &prev_decoded))
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

        Ok(get_new_text(&decoded, &prev_decoded))
    }
}

fn get_new_text(full: &str, prev: &str) -> Option<String> {
    let prev_chars: Vec<char> = prev.chars().collect();
    let full_chars: Vec<char> = full.chars().collect();

    if full_chars.len() > prev_chars.len() {
        let new_chars: String = full_chars[prev_chars.len()..].iter().collect();
        Some(new_chars)
    } else {
        None
    }
}
