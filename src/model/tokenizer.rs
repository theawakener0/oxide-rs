use std::fs::{self, File};
use std::path::PathBuf;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use shimmytok::Tokenizer as ShimmyTokenizer;

const CACHE_DIR: &str = ".cache/oxide";

pub struct TokenizerWrapper {
    inner: ShimmyTokenizer,
    eos_token_id: u32,
    pending_tokens: Vec<u32>,
    cached_decoded: String,
}

fn get_cache_path(model_path: &PathBuf) -> Result<PathBuf> {
    let mut hasher = Sha256::new();

    let mut file = File::open(model_path)?;
    let mut header = vec![0u8; 65536];
    use std::io::Read;
    file.read_exact(&mut header)?;
    hasher.update(&header);

    let file_size = file.metadata()?.len();
    hasher.update(file_size.to_le_bytes());

    let hash = format!("{:x}", hasher.finalize());

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let cache_dir = PathBuf::from(home).join(CACHE_DIR);
    fs::create_dir_all(&cache_dir)?;

    Ok(cache_dir.join(format!("{}.tokenizer_cache", hash)))
}

fn extract_tokenizer_json(path: &PathBuf) -> Result<Option<String>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap);

    let content = gguf_file::Content::read(&mut cursor)
        .map_err(|e| anyhow::anyhow!("Failed to read GGUF: {}", e))?;

    let md = &content.metadata;

    if let Some(tokenizer_json) = md.get("tokenizer.json.json") {
        if let Ok(s) = tokenizer_json.to_string() {
            tracing::debug!("Found embedded tokenizer.json in GGUF");
            return Ok(Some(s.clone()));
        }
    }

    Ok(None)
}

impl TokenizerWrapper {
    pub fn from_gguf(path: &PathBuf) -> Result<Self> {
        let cache_path = get_cache_path(path)?;
        let json_cache_path = cache_path.with_extension("tokenizer_json");

        let inner = if json_cache_path.exists() {
            tracing::info!("Loading tokenizer from JSON cache: {:?}", json_cache_path);
            match ShimmyTokenizer::from_gguf_file(&json_cache_path) {
                Ok(tok) => tok,
                Err(e) => {
                    tracing::warn!("JSON cache corrupted ({}), loading from GGUF", e);
                    ShimmyTokenizer::from_gguf_file(path)?
                }
            }
        } else if cache_path.exists() {
            tracing::info!("Loading tokenizer from legacy cache: {:?}", cache_path);
            ShimmyTokenizer::from_gguf_file(&cache_path)
                .map_err(|e| {
                    tracing::warn!("Legacy cache corrupted ({}), loading from GGUF", e);
                    e
                })
                .or_else(|_| ShimmyTokenizer::from_gguf_file(path))?
        } else {
            tracing::info!("Loading tokenizer from GGUF (first time)...");

            let tokenizer = ShimmyTokenizer::from_gguf_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

            if let Ok(Some(tokenizer_json)) = extract_tokenizer_json(path) {
                if let Err(e) = fs::write(&json_cache_path, &tokenizer_json) {
                    tracing::warn!("Failed to cache tokenizer JSON: {}", e);
                } else {
                    tracing::info!("Tokenizer JSON cached to {:?}", json_cache_path);
                }
            } else if let Err(e) = fs::copy(path, &cache_path) {
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
            cached_decoded: String::new(),
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
            cached_decoded: String::new(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encode failed: {}", e))
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.encode(text)?);
        }
        Ok(results)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, false)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }

    pub fn decode_batch(&self, tokens_batch: &[Vec<u32>]) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(tokens_batch.len());
        for tokens in tokens_batch {
            results.push(self.decode(tokens)?);
        }
        Ok(results)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn is_special_token(&self, token_id: u32) -> bool {
        self.inner.is_special_token(token_id)
    }

    pub fn clear_cache(&mut self) {
        self.pending_tokens.clear();
        self.cached_decoded.clear();
    }

    pub fn decode_next(&mut self, token: u32) -> Result<Option<String>> {
        self.pending_tokens.push(token);

        let new_token_decoded = self.inner.decode_single(token, false)?;

        self.cached_decoded.push_str(&new_token_decoded);

        let result = if new_token_decoded.is_empty() {
            None
        } else {
            Some(new_token_decoded)
        };

        Ok(result)
    }

    pub fn decode_rest(&mut self) -> Result<Option<String>> {
        if self.pending_tokens.is_empty() {
            return Ok(None);
        }

        let result = if self.cached_decoded.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.cached_decoded))
        };

        Ok(result)
    }
}
