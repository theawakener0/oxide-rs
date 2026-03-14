# Getting Started

Oxide is a CLI and Rust library for running GGUF language models locally.

## Requirements

- Rust 1.70+
- A GGUF model file
- For chat mode, a model with an embedded chat template

Oxide is currently focused on CPU-based local inference.

## Install

Install from crates.io:

```bash
cargo install oxide-rs
```

Build from source:

```bash
git clone https://github.com/theawakener0/oxide-rs.git
cd oxide-rs
cargo build --release
```

The release binary is available at `target/release/oxide-rs`.

## Quick start

Download a model from Hugging Face:

```bash
oxide-rs --download "meta-llama/Llama-3.2-1B-Instruct"
```

Inspect a repo before downloading:

```bash
oxide-rs --info "Qwen/Qwen2.5-0.5B-Instruct"
```

List registered local models:

```bash
oxide-rs --models
```

Run interactive chat:

```bash
oxide-rs --model /path/to/model.gguf
```

Run one-shot generation:

```bash
oxide-rs --model /path/to/model.gguf --once --prompt "Hello"
```

Use a custom system prompt:

```bash
oxide-rs --model /path/to/model.gguf --system "You are a Rust expert."
```

Tune sampling:

```bash
oxide-rs --model /path/to/model.gguf \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.15
```

## Supported models

Oxide currently targets:

- GGUF model files
- LLaMA-compatible architectures
- LFM2

For chat-style usage, prefer GGUF models that include an embedded chat template.

## Interactive mode

When running without `--once`, Oxide keeps conversation history and accepts these commands:

| Command | Description |
| --- | --- |
| `/clear` | Clear conversation history |
| `/context` | Show current context usage |
| `/stats` | Show model and generation settings |
| `/help` | Show available commands |
| `/exit` or `/quit` | Exit the program |

## OpenAI-Compatible Server

Run oxide-rs as an OpenAI API-compatible HTTP server:

```bash
oxide-rs --server --port 8080
```

The server starts and listens on the specified port. Models are loaded lazily on first request and cached for subsequent requests.

### Example usage

```bash
# Non-streaming request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model.gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model.gguf",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'

# List loaded models
curl http://localhost:8080/v1/models
```

## Model management

Show model repo details and available GGUF files:

```bash
oxide-rs --info "Qwen/Qwen2.5-0.5B-Instruct"
```

Download a model:

```bash
oxide-rs --download "meta-llama/Llama-3.2-1B-Instruct"
```

List registered models:

```bash
oxide-rs --models
```

Remove a registered model entry:

```bash
oxide-rs --remove "model-id"
```

Downloaded files are stored in the Hugging Face cache at `~/.cache/huggingface/hub/`.
Oxide tracks downloaded models in `~/.oxide/models.json`.

## Common workflows

Run from source during development:

```bash
cargo run --release -- --model /path/to/model.gguf
```

One-shot generation from source:

```bash
cargo run --release -- --model /path/to/model.gguf --once --prompt "Write a hello world program in Rust"
```

Check formatting and build health:

```bash
cargo fmt
cargo check
cargo test
```

## Caveats

- Prefer long flags in examples for clarity.
- Chat-oriented usage expects a GGUF file with an embedded chat template.
- Oxide targets local CPU inference rather than GPU setup.

## Next

- [docs/api-reference.md](docs/api-reference.md) for CLI flags and library surface
- [docs/library-usage.md](docs/library-usage.md) for embedding Oxide in Rust code
- [docs/examples.md](docs/examples.md) for concrete code samples
- [docs/architecture.md](docs/architecture.md) for internals and performance notes
