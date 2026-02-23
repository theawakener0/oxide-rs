# Getting Started with Oxide-rs

Oxide-rs is both a CLI tool and a Rust library for AI inference. This guide covers installation and quick start for both usage modes.

## Installation

### Prerequisites

- Rust 1.70+ (2021 edition)
- A GGUF quantized model file with embedded chat template

### From crates.io (Library/CLI)

```bash
# Add as a dependency to your Rust project
cargo add oxide-rs

# Or add to Cargo.toml manually
oxide-rs = "0.1.0"
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/oxide.git
cd oxide

# Build release binary
make build

# Or using cargo directly
cargo build --release
```

### Install Locally

```bash
make install
# Installs to ~/.local/bin/oxide-rs
```

## CLI Quick Start

```bash
# Interactive chat mode
./target/release/oxide-rs --model ~/Models/your-model-Q4_K_M.gguf

# One-shot generation
./target/release/oxide-rs --model ~/Models/model.gguf --once --prompt "Hello!"

# With custom parameters
./target/release/oxide-rs --model ~/Models/model.gguf \
  --temperature 0.8 \
  --max-tokens 256
```

## Library Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxide-rs = "0.1.0"
```

### Basic Usage

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = GenerateOptions::default();
    let result = generate("model.gguf", options, "Hello, how are you?")?;
    println!("{}", result);
    Ok(())
}
```

### Using the Builder API

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")
        .with_options(oxide_rs::GenerateOptions {
            max_tokens: 256,
            temperature: 0.7,
            ..Default::default()
        })
        .load()?;

    let response = model.generate("What is Rust?")?;
    println!("{}", response);
    Ok(())
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `~/Models/model.gguf` | Path to GGUF model for `make run` |

```bash
export MODEL=~/Models/mistral-7b.Q4_K_M.gguf
make run
```

## Next Steps

- [Library Usage Guide](library-usage.md) - Deep dive into the library API
- [API Reference](api-reference.md) - Detailed API documentation
- [Examples](examples.md) - More code examples
