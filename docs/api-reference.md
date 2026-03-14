# API Reference

Reference for the Oxide CLI and Rust library.

## CLI

### Model management

| Flag | Description |
| --- | --- |
| `--download <repo>` | Download a model from Hugging Face |
| `--models` | List locally registered models |
| `--info <repo>` | Show repository files and recommended GGUF |
| `--remove <id>` | Remove a registered model entry |

### Generation

| Flag | Default | Description |
| --- | --- | --- |
| `--model <path>` | required | Path to a GGUF model file |
| `--tokenizer <path>` | auto | Optional tokenizer path |
| `--system <text>` | none | System prompt |
| `--prompt <text>` | none | Prompt for one-shot mode |
| `--once` | `false` | Run once and exit |
| `--max-tokens <n>` | `512` | Maximum generated tokens |
| `--temperature <f64>` | `0.3` | Sampling temperature |
| `--top-k <n>` | none | Top-k sampling |
| `--top-p <f64>` | none | Nucleus sampling |
| `--repeat-penalty <f32>` | `1.1` | Repeat penalty |
| `--repeat-last-n <n>` | `64` | Repeat penalty window |
| `--batch-size <n>` | `128` | Warmup/prefill batch size |
| `--seed <u64>` | `299792458` | Random seed |
| `--threads <n>` | auto | CPU threads |
| `--max-batch-size <n>` | `8` | Dynamic batching limit |
| `--batch-window-ms <n>` | `100` | Dynamic batching window |
| `--simd <level>` | `auto` | `auto`, `avx512`, `avx2`, `neon`, `scalar` |

Notes:

- CLI defaults shown here are the command-line defaults.
- You can use TUI by typing `--tui`.
- Library defaults for `GenerateOptions` differ for some batching-related fields because they come from the crate API rather than the CLI wrapper.

### Interactive commands

| Command | Description |
| --- | --- |
| `/clear` | Clear conversation history |
| `/context` | Show current context usage |
| `/stats` | Show model info and current settings |
| `/help` | Show available commands |
| `/exit` or `/quit` | Exit interactive mode |

## Library

### `generate`

Simple one-shot generation.

```rust
pub fn generate<P: AsRef<Path>>(
    model_path: P,
    options: GenerateOptions,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>>
```

Example:

```rust
use oxide_rs::{generate, GenerateOptions};

let output = generate("model.gguf", GenerateOptions::default(), "Hello")?;
```

### `GenerateOptions`

Configuration for generation.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_tokens` | `usize` | `512` | Maximum generated tokens |
| `temperature` | `f64` | `0.3` | Sampling temperature |
| `top_p` | `Option<f64>` | `None` | Nucleus sampling threshold |
| `top_k` | `Option<usize>` | `None` | Top-k sampling threshold |
| `repeat_penalty` | `f32` | `1.1` | Repeat penalty |
| `repeat_last_n` | `usize` | `64` | Repeat penalty window |
| `batch_size` | `usize` | `128` | Warmup/prefill batch size |
| `seed` | `u64` | `299792458` | Random seed |
| `system_prompt` | `Option<String>` | `None` | Optional system prompt |
| `max_batch_size` | `usize` | `4` | Dynamic batching limit |
| `batch_window_ms` | `u64` | `1` | Dynamic batching window |
| `enable_prefix_cache` | `bool` | `true` | Enable prefix caching |
| `cache_memory_mb` | `usize` | `512` | Prefix cache memory budget |
| `cpu_threads` | `usize` | `0` | CPU threads, `0` means auto |
| `reserve_cores` | `usize` | `0` | CPU cores reserved for OS |
| `simd_level` | `String` | `"auto"` | SIMD level selection |

Example:

```rust
use oxide_rs::GenerateOptions;

let options = GenerateOptions {
    max_tokens: 256,
    temperature: 0.7,
    top_p: Some(0.9),
    top_k: Some(40),
    repeat_penalty: 1.15,
    ..Default::default()
};
```

### `Model`

High-level wrapper for repeated generations.

```rust
pub struct Model {
    // private fields
}
```

Core methods:

| Method | Purpose |
| --- | --- |
| `Model::new(path)` | Create a model handle |
| `with_options(options)` | Set generation options |
| `with_tokenizer(path)` | Use a custom tokenizer |
| `load()` | Load the model into memory |
| `generate(prompt)` | Generate a full response |
| `generate_stream(prompt, callback)` | Stream tokens as they are produced |
| `generate_batch(prompts)` | Generate for multiple prompts |
| `warmup(num_tokens)` | Warm up compute paths |
| `clear_history()` | Clear conversation history |
| `metadata()` | Access GGUF metadata |
| `context_used()` | Current context usage |
| `context_limit()` | Maximum context window |
| `context_percentage()` | Context usage as a percentage |

Example:

```rust
use oxide_rs::Model;

let mut model = Model::new("model.gguf")?
    .with_options(oxide_rs::GenerateOptions {
        max_tokens: 256,
        temperature: 0.7,
        ..Default::default()
    });

model.load()?;

let output = model.generate("Explain ownership in Rust.")?;
```

### `StreamEvent`

Streaming generation emits these events:

```rust
pub enum StreamEvent {
    Token(String),
    PrefillStatus(usize),
    Done,
}
```

### `GgufMetadata`

Metadata extracted from the model file.

```rust
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

## More

- [docs/getting-started.md](docs/getting-started.md) for CLI workflows
- [docs/library-usage.md](docs/library-usage.md) for deeper library examples
- [docs/examples.md](docs/examples.md) for sample programs
- [docs/architecture.md](docs/architecture.md) for implementation details
