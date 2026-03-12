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

- `docs/getting-started.md` - installation, CLI workflows, model management
- `docs/api-reference.md` - CLI flags, interactive commands, library API
- `docs/library-usage.md` - embedding Oxide in Rust code
- `docs/examples.md` - usage patterns and code snippets
- `docs/architecture.md` - internals, supported models, performance notes

## License

MIT. See [LICENSE](LICENSE).
