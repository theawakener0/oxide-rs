# Oxide

**Fast AI Inference CLI in Rust** — A lightweight, CPU-based LLM inference engine inspired by llama.cpp.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **GGUF Model Support** — Load quantized models in GGUF format
- **Full Tokenizer Compatibility** — Supports all llama.cpp tokenizer types via [shimmytok](https://crates.io/crates/shimmytok) (SPM, BPE, WPM, UGM, RWKV)
- **Automatic Chat Templates** — Uses Jinja templates embedded in GGUF files via [minijinja](https://crates.io/crates/minijinja)
- **Streaming Output** — Real-time token generation with tokens-per-second metrics
- **Multiple Sampling Strategies** — Temperature, top-k, top-p, and argmax sampling
- **Repeat Penalty** — Prevents repetitive output with configurable penalty window
- **Interactive REPL** — Full conversation mode with session history
- **One-Shot Mode** — Non-interactive generation for scripting/pipelines
- **Beautiful CLI** — Animated loading, syntax-highlighted output, Rust-themed

## Installation

### Prerequisites

- Rust 1.70+ (2021 edition)
- A GGUF quantized model file with embedded chat template

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
# Installs to ~/.local/bin/oxide
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `~/Models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | Path to GGUF model for `make run` |

```bash
# Set custom model path for make run
export MODEL=~/Models/mistral-7b-v0.1.Q4_K_M.gguf
make run

# Or inline
MODEL=~/Models/phi-3.Q4_K_M.gguf make run
```

## Quick Start

```bash
# Interactive chat mode
./target/release/oxide --model ~/Models/your-model-Q4_K_M.gguf

# With custom system prompt
./target/release/oxide --model ~/Models/model.gguf --system "You are a Rust expert."

# One-shot generation
./target/release/oxide --model ~/Models/model.gguf --once --prompt "Write a Rust function to reverse a string"

# With custom sampling parameters
./target/release/oxide --model ~/Models/model.gguf \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.15
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | *required* | Path to GGUF model file |
| `-t, --tokenizer` | *auto* | Path to tokenizer.json (extracted from GGUF if omitted) |
| `-s, --system` | *none* | System prompt for the model |
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (0.0 = greedy/argmax) |
| `--top-k` | *none* | Top-k sampling threshold |
| `--top-p` | *none* | Nucleus sampling threshold |
| `--repeat-penalty` | `1.1` | Penalty for repeated tokens |
| `--repeat-last-n` | `64` | Context window for repeat penalty |
| `--seed` | `299792458` | Random seed for reproducibility |
| `-p, --prompt` | *none* | Input prompt (for one-shot mode) |
| `-o, --once` | `false` | Run in non-interactive mode |

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history for current session |
| `/exit` or `/quit` | Exit the program |
| `/help` | Show available commands |

## Chat Templates

Oxide automatically uses the chat template embedded in GGUF files:

```mermaid
graph LR
    A[GGUF File] --> B[tokenizer.chat_template]
    B --> C[minijinja]
    C --> D[Rendered Prompt]
```

### How It Works

1. **Extraction** — Reads `tokenizer.chat_template` from GGUF metadata
2. **Rendering** — Uses [minijinja](https://crates.io/crates/minijinja) to render Jinja2 templates
3. **Multi-turn** — Maintains conversation history within the session

### Example Template (ChatML)

```jinja2
{% for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}
<|im_start|>assistant
```

### Supported Models

Any GGUF model with an embedded `tokenizer.chat_template` will work automatically:

| Model Family | Template Source |
|--------------|-----------------|
| LLaMA 3.x | Embedded in GGUF |
| Mistral | Embedded in GGUF |
| Qwen | Embedded in GGUF |
| Gemma | Embedded in GGUF |
| Phi-3 | Embedded in GGUF |
| SmolLM | Embedded in GGUF |
| LFM | Embedded in GGUF |

> **Note**: If your GGUF file lacks a chat template, Oxide will error and ask you to use a model with an embedded template.

## Supported Models

### Model Architectures

Oxide uses [Candle](https://github.com/huggingface/candle) for inference:

- **LLaMA** — LLaMA 2, LLaMA 3, Mistral, Qwen, Phi, etc.
- **LFM2** — Liquid Foundation Models

> **Note**: Any GGUF model with LLaMA-compatible or LFM2 architecture should work.

### Tokenizer Support

Oxide uses [shimmytok](https://crates.io/crates/shimmytok) for tokenizer support, providing 100% llama.cpp compatibility:

| Tokenizer Type | Description |
|----------------|-------------|
| **SPM** | SentencePiece (LLaMA, Mistral, etc.) |
| **BPE** | Byte-Pair Encoding (GPT-2 style) |
| **WPM** | WordPiece Model (BERT style) |
| **UGM** | Unigram Model |
| **RWKV** | RWKV tokenizers |

## Architecture

```mermaid
graph TB
    subgraph CLI["CLI Layer (clap)"]
        args[Argument Parser]
    end

    subgraph TUI["Terminal UI (crossterm)"]
        banner[Animated Banner]
        loader[Ferris Loader]
        stream[Streaming Output]
        theme[Color Theme]
    end

    subgraph Inference["Inference Layer"]
        generator[Generator]
        template[Chat Template]
        sampler[Logits Processor]
        callback[StreamEvent Callback]
    end

    subgraph Model["Model Layer"]
        model[Model Weights]
        tokenizer[Tokenizer Wrapper]
    end

    subgraph Core["Core Dependencies"]
        candle[Candle Transformers]
        shimmytok[shimmytok]
        minijinja[minijinja]
    end

    args --> generator
    generator --> template
    generator --> sampler
    generator --> model
    generator --> tokenizer
    generator --> callback
    
    callback --> stream
    template --> minijinja
    
    model --> candle
    tokenizer --> shimmytok
    
    banner --> theme
    loader --> theme
    stream --> theme
```

### Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Generator
    participant Template
    participant Model
    participant Tokenizer
    participant Stream

    User->>CLI: Enter prompt
    CLI->>Generator: generate(prompt, config)
    Generator->>Template: apply(messages)
    Template-->>Generator: formatted prompt
    Generator->>Tokenizer: encode(prompt)
    Tokenizer-->>Generator: tokens[]
    Generator->>Model: forward(tokens)
    Model-->>Generator: logits
    
    loop For each token
        Generator->>Generator: sample(logits)
        Generator->>Tokenizer: decode_next(token)
        Tokenizer-->>Generator: text
        Generator->>Stream: StreamEvent::Token(text)
        Stream->>User: Display token
    end
    
    Generator->>Stream: StreamEvent::Done
    Stream->>User: Show stats (tok/s)
```

## Development

```bash
# Development build (faster compile)
make dev

# Run with model (uses MODEL env var)
make run

# Or override MODEL inline
make run MODEL=~/path/to/model.gguf

# Format code
make fmt

# Run linter
make check

# Clean build artifacts
make clean
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor operations, ML primitives |
| `candle-nn` | Neural network layers |
| `candle-transformers` | Pre-built model architectures (LLaMA, LFM2) |
| `shimmytok` | GGUF tokenizer (100% llama.cpp compatible) |
| `minijinja` | Jinja2 template engine for chat templates |
| `clap` | CLI argument parsing with derive macros |
| `crossterm` | Cross-platform terminal control |
| `anyhow` | Ergonomic error handling |
| `serde` | Serialization for messages |
| `tracing` | Structured logging |

## Performance

- **CPU-only inference** — No GPU dependencies, portable binaries
- **Quantized models** — Q4_K_M provides good quality/speed tradeoff; other quantizations supported
- **Streaming decode** — Tokens displayed as generated for responsive UX
- **Context caching** — Efficient multi-turn conversations with token history management

## Roadmap

- [ ] Multi-modal support
- [ ] OpenAI-compatible API server
- [ ] Model download/management

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) — HuggingFace's minimalist ML framework for Rust
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Inspiration and GGUF format specification
- [shimmytok](https://crates.io/crates/shimmytok) — Pure Rust GGUF tokenizer with llama.cpp compatibility
- [minijinja](https://crates.io/crates/minijinja) — Minimal Jinja2 template engine for Rust
