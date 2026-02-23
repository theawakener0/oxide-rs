//! Oxide-rs
//!
//! Fast AI Inference Library & CLI in Rust - A lightweight, CPU-based LLM inference engine inspired by llama.cpp.
//!
//! # Features
//!
//! - GGUF model support (LLaMA, LFM2 architectures)
//! - Full tokenizer compatibility (SPM, BPE, WPM, UGM, RWKV)
//! - Automatic chat templates from GGUF files
//! - Streaming token generation
//! - Multiple sampling strategies (temperature, top-k, top-p)
//! - Interactive REPL and one-shot modes
//! - Memory-mapped loading for instant startup
//!
//! # Quick Start
//!
//! ## CLI Usage
//!
//! ```bash
//! # Install via cargo
//! cargo install oxide-rs
//!
//! # Run interactively
//! oxide-rs -m model.gguf
//!
//! # One-shot generation
//! oxide-rs -m model.gguf --once --prompt "Hello!"
//! ```
//!
//! ## Library Usage
//!
//! ```rust,ignore
//! use oxide_rs::{generate, GenerateOptions};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let result = generate(
//!         "model.gguf",
//!         GenerateOptions::default(),
//!         "Hello, how are you?",
//!     )?;
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```
//!
//! ## Builder API
//!
//! For more control, use the `Model` builder:
//!
//! ```rust,ignore
//! use oxide_rs::Model;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut model = Model::new("model.gguf")
//!         .with_options(oxide_rs::GenerateOptions {
//!             max_tokens: 256,
//!             temperature: 0.7,
//!             ..Default::default()
//!         })
//!         .load()?;
//!
//!     let response = model.generate("What is Rust?")?;
//!     println!("{}", response);
//!     Ok(())
//! }
//! ```
//!
//! # Requirements
//!
//! - Rust 1.70+ (2021 edition)
//! - A GGUF quantized model file with embedded chat template
//!
//! # Links
//!
//! - [GitHub Repository](https://github.com/theawakener0/oxide-rs)
//! - [crates.io](https://crates.io/crates/oxide-rs)
//! - [Documentation](https://docs.rs/oxide-rs)

pub mod cli;
pub mod inference;
pub mod model;

use std::path::Path;
use std::path::PathBuf;

pub use inference::{Generator, StreamEvent};
pub use model::{GgufMetadata, Model as ModelWrapper, TokenizerWrapper};

/// Configuration options for text generation.
///
/// # Example
///
/// ```rust,ignore,ignore
/// use oxide_rs::GenerateOptions;
///
/// let options = GenerateOptions {
///     max_tokens: 512,
///     temperature: 0.3,
///     top_p: None,
///     top_k: None,
///     repeat_penalty: 1.1,
///     repeat_last_n: 64,
///     seed: 299792458,
///     system_prompt: None,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    /// Maximum number of tokens to generate.
    ///
    /// Default: `512`
    pub max_tokens: usize,

    /// Sampling temperature. Higher values produce more diverse output,
    /// lower values produce more focused output.
    ///
    /// Set to `0.0` for greedy/argmax sampling.
    ///
    /// Default: `0.3`
    pub temperature: f64,

    /// Nucleus sampling (top-p) threshold. Limits sampling to the smallest
    /// set of tokens whose cumulative probability exceeds this threshold.
    ///
    /// Default: `None`
    pub top_p: Option<f64>,

    /// Top-k sampling. Limits sampling to the k most likely tokens.
    ///
    /// Default: `None`
    pub top_k: Option<usize>,

    /// Penalty applied to repeated tokens. Values > 1.0 reduce repetition.
    ///
    /// Default: `1.1`
    pub repeat_penalty: f32,

    /// Number of previous tokens to consider for repeat penalty.
    ///
    /// Default: `64`
    pub repeat_last_n: usize,

    /// Random seed for reproducibility. Same seed + same input = same output.
    ///
    /// Default: `299792458`
    pub seed: u64,

    /// System prompt to prepend to the conversation.
    ///
    /// Default: `None`
    pub system_prompt: Option<String>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.3,
            top_p: None,
            top_k: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 299792458,
            system_prompt: None,
        }
    }
}

/// High-level model wrapper with builder pattern for text generation.
///
/// Use this when you need to:
/// - Generate multiple times with the same model
/// - Use streaming callbacks
/// - Maintain conversation history
/// - Access model metadata
///
/// # Example
///
/// ```rust,ignore,ignore
/// use oxide_rs::Model;
///
/// let mut model = Model::new("model.gguf")?
///     .with_options(oxide_rs::GenerateOptions {
///         max_tokens: 256,
///         temperature: 0.7,
///         ..Default::default()
///     })
///     .load()?;
///
/// let response = model.generate("Hello!")?;
/// println!("{}", response);
/// ```
pub struct Model {
    generator: Option<Generator>,
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    options: GenerateOptions,
}

impl Model {
    /// Create a new Model instance.
    ///
    /// This only creates the Model struct - use `load()` to actually load the model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to a GGUF model file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Model::new("model.gguf")?;
    /// ```
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            generator: None,
            model_path: model_path.as_ref().to_path_buf(),
            tokenizer_path: None,
            options: GenerateOptions::default(),
        })
    }

    /// Set generation options.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Model::new("model.gguf")
    ///     .with_options(GenerateOptions {
    ///         max_tokens: 256,
    ///         temperature: 0.8,
    ///         ..Default::default()
    ///     });
    /// ```
    pub fn with_options(mut self, options: GenerateOptions) -> Self {
        self.options = options;
        self
    }

    /// Set a custom tokenizer path.
    ///
    /// If not provided, the tokenizer will be extracted from the GGUF file.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Model::new("model.gguf")
    ///     .with_tokenizer("tokenizer.json");
    /// ```
    pub fn with_tokenizer<P: AsRef<Path>>(mut self, tokenizer_path: P) -> Self {
        self.tokenizer_path = Some(tokenizer_path.as_ref().to_path_buf());
        self
    }

    /// Load the model into memory.
    ///
    /// This must be called before `generate()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut model = Model::new("model.gguf")?.load()?;
    /// ```
    pub fn load(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let generator = Generator::new(
            &self.model_path,
            self.tokenizer_path.as_ref(),
            self.options.temperature,
            self.options.top_p,
            self.options.top_k,
            self.options.seed,
            self.options.system_prompt.clone(),
        )?;
        self.generator = Some(generator);
        Ok(())
    }

    /// Generate text from a prompt.
    ///
    /// Requires `load()` to be called first.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input prompt
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = model.generate("What is Rust?")?;
    /// println!("{}", response);
    /// ```
    pub fn generate(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let generator = self
            .generator
            .as_mut()
            .ok_or("Model not loaded. Call load() first.")?;

        let result = generator.generate(
            prompt,
            self.options.max_tokens,
            self.options.repeat_penalty,
            self.options.repeat_last_n,
            |_event| {},
        )?;

        Ok(result)
    }

    /// Generate text with streaming callback.
    ///
    /// Tokens are passed to the callback as they're generated, enabling
    /// real-time output display.
    ///
    /// Requires `load()` to be called first.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input prompt
    /// * `callback` - Function called for each generated token
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.generate_stream("Tell me a story", |token| {
    ///     print!("{}", token);
    /// })?;
    /// ```
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        mut callback: F,
    ) -> Result<String, Box<dyn std::error::Error>>
    where
        F: FnMut(String),
    {
        let generator = self
            .generator
            .as_mut()
            .ok_or("Model not loaded. Call load() first.")?;

        let mut output = String::new();
        generator.generate(
            prompt,
            self.options.max_tokens,
            self.options.repeat_penalty,
            self.options.repeat_last_n,
            |event| match event {
                StreamEvent::Token(t) => {
                    output.push_str(&t);
                    callback(t);
                }
                StreamEvent::Done => {}
            },
        )?;

        Ok(output)
    }

    /// Pre-compile compute kernels for faster first-token generation.
    ///
    /// Call this after `load()` to warm up the model before first use.
    ///
    /// # Arguments
    ///
    /// * `num_tokens` - Number of tokens to use for warmup (default: 128)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.load()?;
    /// model.warmup(128)?;
    /// // First generation will be faster
    /// ```
    pub fn warmup(&mut self, num_tokens: usize) -> Result<(), Box<dyn std::error::Error>> {
        let generator = self
            .generator
            .as_mut()
            .ok_or("Model not loaded. Call load() first.")?;
        generator.warmup(num_tokens)?;
        Ok(())
    }

    /// Clear conversation history.
    ///
    /// Removes all previous messages from the conversation context.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.generate("Hello")?;
    /// model.clear_history();
    /// ```
    pub fn clear_history(&mut self) {
        if let Some(ref mut generator) = self.generator {
            generator.clear_history();
        }
    }

    /// Get model metadata.
    ///
    /// Returns information about the loaded model including name,
    /// architecture, layer count, embedding size, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(meta) = model.metadata() {
    ///     println!("Model: {}", meta.name);
    ///     println!("Architecture: {}", meta.architecture);
    /// }
    /// ```
    pub fn metadata(&self) -> Option<&GgufMetadata> {
        self.generator.as_ref().map(|g| g.metadata())
    }
}

/// Simple one-shot text generation function.
///
/// This is the easiest way to generate text - just provide the model path,
/// options, and prompt. The model is loaded and used in a single call.
///
/// For multiple generations, use [`Model`] instead to avoid reloading.
///
/// # Arguments
///
/// * `model_path` - Path to GGUF model file
/// * `options` - Generation configuration
/// * `prompt` - Input prompt
///
/// # Returns
///
/// Generated text string
///
/// # Example
///
/// ```rust,ignore,ignore
/// use oxide_rs::{generate, GenerateOptions};
///
/// let result = generate(
///     "model.gguf",
///     GenerateOptions::default(),
///     "Hello, how are you?",
/// )?;
/// println!("{}", result);
/// ```
pub fn generate<P: AsRef<Path>>(
    model_path: P,
    options: GenerateOptions,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut model = Model::new(model_path)?.with_options(options);
    model.load()?;
    model.generate(prompt)
}
