# oxide-rs

A high-performance, memory-safe, and lightweight LLM inference engine written in pure Rust. Optimized for CPU-based inference and inspired by the efficiency of llama.cpp.

## Why oxide-rs

- Run GGUF models locally with a simple CLI.
- Stream tokens in real time.
- Use embedded GGUF chat templates automatically.
- Download and track models from Hugging Face.
- Embed the same inference stack in Rust applications.

## Install

```bash
cargo install oxide-rs
```

Or build from source:

```bash
git clone https://github.com/theawakener0/oxide-rs.git
cd oxide-rs
cargo build --release
```

## Quick start

Download a GGUF model:

```bash
oxide-rs --download "meta-llama/Llama-3.2-1B-Instruct"
```

List downloaded models:

```bash
oxide-rs --models
```

Run interactive chat:

```bash
oxide-rs --model /path/to/model.gguf
```

Run one-shot generation:

```bash
oxide-rs --model /path/to/model.gguf --once --prompt "Write a Rust function to reverse a string"
```

Adjust sampling:

```bash
oxide-rs --model /path/to/model.gguf --temperature 0.8 --top-k 40 --top-p 0.9
```

## OpenAI-Compatible Server

Run oxide-rs as an OpenAI API-compatible HTTP server:

```bash
oxide-rs --server --port 8080
```

The server provides OpenAI-compatible endpoints:

```bash
# Chat completions (non-streaming)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model.gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Chat completions (streaming)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# List available models
curl http://localhost:8080/v1/models
```

Features:
- Specify model path in request body (lazy loading)
- Models are cached after first use
- Streaming support with Server-Sent Events
- OpenAI-compatible response format
- CORS enabled for browser clients

## TUI (Terminal UI)

The TUI provides an interactive sidebar-driven interface with:

- Chat screen with live streaming and thinking spinner
- Models screen with selection and active/highlighted markers
- Settings screen for generation parameters and system prompt editing

Run the TUI:

```bash
oxide-rs --tui --model /path/to/model.gguf
```

Shortcuts:

- `F1` or `?` — toggle help/shortcuts overlay
- `Tab` / `Shift+Tab` — cycle focus between sidebar, main panel, and input
- Sidebar: `j` / `k` to select, `Enter` to open a screen
- Chat: `Enter` to send prompt; `j`/`k` scrolls history when main panel focused
- Models: `j`/`k` to move, `Enter` to load, `x` to remove, `d` shows download hint
- Settings: `j`/`k` to choose field, `h`/`l` to adjust, `Enter` to apply, `r` to reset, type to edit system prompt

## Common commands

```bash
# show files before downloading
oxide-rs --info "Qwen/Qwen2.5-0.5B-Instruct"

# remove a registered model
oxide-rs --remove "model-id"
```

## Requirements

- Rust 1.70+
- A GGUF model file
- For chat mode, use a model with an embedded chat template

Oxide is currently focused on CPU-based local inference.

## Supported formats

- GGUF
- LLaMA-compatible architectures
- LFM2

## Library

```bash
cargo add oxide-rs
```

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = generate(
        "model.gguf",
        GenerateOptions::default(),
        "Explain what Rust ownership means.",
    )?;

    println!("{}", output);
    Ok(())
}
```

## Docs

- [docs/getting-started.md](docs/getting-started.md) - installation, CLI workflows, model management
- [docs/api-reference.md](docs/api-reference.md) - CLI flags, interactive commands, library API
- [docs/library-usage.md](docs/library-usage.md) - embedding Oxide in Rust code
- [docs/examples.md](docs/examples.md) - usage patterns and code snippets
- [docs/architecture.md](docs/architecture.md) - internals, supported models, performance notes

## License

MIT. See [LICENSE](LICENSE).
