# API Reference

Complete API documentation for oxide-rs.

## Functions

### `generate`

Simple one-shot generation function.

```rust
pub fn generate<P: AsRef<Path>>(
    model_path: P,
    options: GenerateOptions,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>>
```

**Parameters:**
- `model_path` - Path to GGUF model file
- `options` - Generation configuration
- `prompt` - Input prompt

**Returns:** Generated text string

**Example:**

```rust
use oxide_rs::{generate, GenerateOptions};

let result = generate("model.gguf", GenerateOptions::default(), "Hello!")?;
```

---

## Structs

### `GenerateOptions`

Configuration for text generation.

```rust
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub system_prompt: Option<String>,
}
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | `usize` | `512` | Maximum tokens to generate |
| `temperature` | `f64` | `0.3` | Sampling temperature (0.0 = greedy/argmax) |
| `top_p` | `Option<f64>` | `None` | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | `Option<usize>` | `None` | Top-k sampling threshold |
| `repeat_penalty` | `f32` | `1.1` | Penalty for repeated tokens (1.0 = no penalty) |
| `repeat_last_n` | `usize` | `64` | Context window for repeat penalty |
| `seed` | `u64` | `299792458` | Random seed for reproducibility |
| `system_prompt` | `Option<String>` | `None` | System prompt to prepend |

**Example:**

```rust
use oxide_rs::GenerateOptions;

let options = GenerateOptions {
    max_tokens: 256,
    temperature: 0.7,
    top_p: Some(0.9),
    top_k: Some(40),
    repeat_penalty: 1.15,
    repeat_last_n: 64,
    seed: 42,
    system_prompt: Some("You are a helpful assistant.".into()),
};
```

### `Model`

High-level model wrapper with builder pattern.

```rust
pub struct Model {
    // private fields
}
```

**Methods:**

#### `new`

Create a new Model instance.

```rust
pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>>
```

**Example:**

```rust
let model = Model::new("model.gguf")?;
```

---

#### `with_options`

Set generation options.

```rust
pub fn with_options(mut self, options: GenerateOptions) -> Self
```

**Example:**

```rust
let model = Model::new("model.gguf")
    .with_options(GenerateOptions {
        max_tokens: 256,
        ..Default::default()
    });
```

---

#### `with_tokenizer`

Set custom tokenizer path.

```rust
pub fn with_tokenizer<P: AsRef<Path>>(self, tokenizer_path: P) -> Self
```

**Example:**

```rust
let model = Model::new("model.gguf")
    .with_tokenizer("tokenizer.json");
```

---

#### `load`

Load the model into memory.

```rust
pub fn load(&mut self) -> Result<(), Box<dyn std::error::Error>>
```

**Example:**

```rust
let mut model = Model::new("model.gguf")?.load()?;
```

---

#### `generate`

Generate text from a prompt.

```rust
pub fn generate(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>>
```

**Example:**

```rust
let response = model.generate("Hello!")?;
```

---

#### `generate_stream`

Generate text with streaming callback.

```rust
pub fn generate_stream<F>(&mut self, prompt: &str, callback: F) -> Result<String, Box<dyn std::error::Error>>
where
    F: FnMut(String),
```

**Example:**

```rust
model.generate_stream("Tell me a story", |token| {
    print!("{}", token);
})?;
```

---

#### `warmup`

Pre-compile compute kernels.

```rust
pub fn warmup(&mut self, num_tokens: usize) -> Result<(), Box<dyn std::error::Error>>
```

**Example:**

```rust
model.warmup(128)?;  // Warmup with 128 tokens
```

---

#### `clear_history`

Clear conversation history.

```rust
pub fn clear_history(&mut self)
```

**Example:**

```rust
model.clear_history();
```

---

#### `metadata`

Get model metadata.

```rust
pub fn metadata(&self) -> Option<&GgufMetadata>
```

**Example:**

```rust
if let Some(meta) = model.metadata() {
    println!("{}", meta.name);
}
```

---

#### `context_used`

Get current context usage (number of tokens in context).

```rust
pub fn context_used(&self) -> Option<usize>
```

**Example:**

```rust
let used = model.context_used().unwrap_or(0);
println!("Using {} tokens", used);
```

---

#### `context_limit`

Get maximum context window size.

```rust
pub fn context_limit(&self) -> Option<usize>
```

**Example:**

```rust
let limit = model.context_limit().unwrap_or(4096);
println!("Context limit: {} tokens", limit);
```

---

#### `context_percentage`

Get context usage as percentage (0.0 - 100.0).

```rust
pub fn context_percentage(&self) -> Option<f32>
```

**Example:**

```rust
let pct = model.context_percentage().unwrap_or(0.0);
println!("{:.1}% context used", pct);
```

---

## Re-exports

These types are also exported at the crate root:

```rust
pub use inference::{Generator, StreamEvent, ChatTemplate, Message};
pub use model::{GgufMetadata, TokenizerWrapper};
```

### `GgufMetadata`

Model metadata extracted from GGUF file.

```rust
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
}
```

### `StreamEvent`

Events during streaming generation.

```rust
pub enum StreamEvent {
    Token(String),
    PrefillStatus(usize),
    Done,
}
```

**Variants:**
- `Token(String)` - A generated token
- `PrefillStatus(usize)` - Prompt processing status (token count)
- `Done` - Generation complete

### `Message`

Chat message structure.

```rust
#[derive(Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}
```

### `ChatTemplate`

Chat template handler.

```rust
pub struct ChatTemplate { /* private */ }

impl ChatTemplate {
    pub fn new(template: Option<String>) -> Result<Self, Box<dyn std::error::Error>>
    pub fn apply(&self, messages: &[Message]) -> Result<String, Box<dyn std::error::Error>>
}
```

---

## Traits

### `Default` for `GenerateOptions`

Provides default configuration values.

```rust
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
```
