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
oxide-rs = "0.1.13"
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/theawakener0/oxide-rs.git
cd oxide-rs

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

### Install via Cargo (CLI)

```bash
cargo install oxide-rs
# Installs to ~/.cargo/bin/oxide-rs
```

## CLI Quick Start

```bash
# If installed via cargo install
oxide-rs --model ~/Models/your-model-Q4_K_M.gguf

# Or run directly from source
./target/release/oxide-rs --model ~/Models/your-model-Q4_K_M.gguf

# One-shot generation
./target/release/oxide-rs --model ~/Models/model.gguf --once --prompt "Hello!"

# With custom parameters
./target/release/oxide-rs --model ~/Models/model.gguf \
  --temperature 0.8 \
  --max-tokens 256

# With performance tuning
./target/release/oxide-rs --model ~/Models/model.gguf \
  --batch-size 256
```

## Interactive Mode Commands

When running in interactive mode, you can use these commands:

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/context` | Show context usage (tokens used / limit) |
| `/stats` | Show model info and current settings |
| `/help` | Show available commands |
| `/exit` | Exit the program |

## Features

- **Thinking Spinner**: Shows `🦀💭 Thinking...` animation while waiting for the first token
- **Live Stats**: Displays tokens per second every 0.5s during generation
- **Context Tracking**: Shows context usage in stats (e.g., `Context: 2048/4096`)
- **Special Token Handling**: Automatically strips chat template tokens and converts newlines
- **Performance Tuning**: Configurable batch size and prefetch size

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | required | Path to GGUF model file |
| `-t, --tokenizer` | auto | Path to tokenizer.json |
| `-s, --system` | auto | System prompt |
| `--max-tokens` | 512 | Maximum tokens to generate |
| `--temperature` | 0.3 | Sampling temperature |
| `--top-k` | none | Top-k sampling |
| `--top-p` | none | Top-p sampling |
| `--repeat-penalty` | 1.1 | Repeat penalty |
| `--repeat-last-n` | 64 | Context window for repeat penalty |
| `--batch-size` | 128 | Batch size for warmup |
| `--seed` | 299792458 | Random seed |
| `--threads` | auto | Thread count |
| `-p, --prompt` | none | Input prompt |
| `-o, --once` | false | Non-interactive mode |

## Library Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxide-rs = "0.1.13"
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
