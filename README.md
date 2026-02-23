# Oxide

**Fast AI Inference CLI in Rust** — A lightweight, CPU-based LLM inference engine inspired by llama.cpp.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **GGUF Model Support** — Load quantized models in GGUF format (LLaMA, LFM2 architectures)
- **Streaming Output** — Real-time token generation with tokens-per-second metrics
- **Multiple Sampling Strategies** — Temperature, top-k, top-p, and argmax sampling
- **Repeat Penalty** — Prevents repetitive output with configurable penalty window
- **Chat Templates** — Automatic prompt formatting for SmolLM, LFM, Gemma, Mistral, Zephyr, Phi
- **Interactive REPL** — Full conversation mode with history persistence
- **One-Shot Mode** — Non-interactive generation for scripting/pipelines
- **Beautiful TUI** — Animated loading, syntax-highlighted output, terminal theming

## Installation

### Prerequisites

- Rust 1.70+ (2021 edition)
- A GGUF quantized model file

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

## Quick Start

```bash
# Interactive chat mode
./target/release/oxide --model ~/Models/your-model-Q4_K_M.gguf

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
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (0.0 = greedy/argmax) |
| `--top-k` | *none* | Top-k sampling threshold |
| `--top-p` | *none* | Nucleus sampling threshold |
| `--repeat-penalty` | `1.1` | Penalty for repeated tokens |
| `--repeat-last-n` | `64` | Context window for repeat penalty |
| `--seed` | `299792458` | Random seed for reproducibility |
| `--max-history` | `2048` | Maximum conversation history in tokens |
| `-p, --prompt` | *none* | Input prompt (for one-shot mode) |
| `-o, --once` | `false` | Run in non-interactive mode |
| `--clear-history` | `false` | Clear saved conversation history |

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/exit` or `/quit` | Exit the program |
| `/help` | Show available commands |

## Supported Models

### Architectures

| Architecture | Status |
|--------------|--------|
| LLaMA | ✅ Supported |
| LFM2 | ✅ Supported |

### Chat Template Compatibility

Oxide automatically detects and applies the correct chat template based on model name:

| Model Family | Template Format |
|--------------|-----------------|
| SmolLM | `<\|im_start\|>user\n...<\|im_end\|>` |
| LFM | `<\|im_start\|>user\n...<\|im_end\|>` |
| Gemma | `<start_of_turn>user\n...<end_of_turn>` |
| Mistral / Zephyr | `[INST] ... [/INST]` |
| Phi | `user\n...<\|end\|>` |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (clap)                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌───────────┐  │
│  │ Banner  │  │ Loader   │  │ Stream  │  │  History  │  │
│  │ (TUI)   │  │ (TUI)    │  │ (TUI)   │  │ (JSON)    │  │
│  └─────────┘  └──────────┘  └─────────┘  └───────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Inference Layer                       │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Generator                                          │ │
│  │  - ChatTemplate (prompt formatting)                 │ │
│  │  - LogitsProcessor (sampling strategies)            │ │
│  │  - StreamEvent (callback-based streaming)           │ │
│  └────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                     Model Layer                          │
│  ┌───────────────┐  ┌────────────────────────────────┐  │
│  │  Model        │  │  TokenizerWrapper              │  │
│  │  (LLaMA/LFM2) │  │  (shimmytok/GGUF extraction)   │  │
│  └───────────────┘  └────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Candle (HuggingFace) + Crossterm            │
└─────────────────────────────────────────────────────────┘
```

## Development

```bash
# Development build (faster compile)
make dev

# Run with model
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
| `candle-transformers` | Pre-built model architectures |
| `tokenizers` | HuggingFace tokenizers |
| `shimmytok` | GGUF tokenizer extraction |
| `clap` | CLI argument parsing |
| `crossterm` | Terminal control (colors, cursor) |
| `anyhow` | Error handling |
| `serde_json` | History serialization |
| `tracing` | Logging |

## Performance

- **CPU-only inference** — No GPU dependencies
- **Quantized models** — Q4_K_M quantization provides good quality/speed tradeoff
- **Streaming decode** — Tokens displayed as generated, not batched
- **Context caching** — KV-cache for efficient multi-turn conversations

## Roadmap

- [ ] Multi-modal support
- [ ] OpenAI-compatible API server
- [ ] Model download/management

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) — HuggingFace's minimalist ML framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Inspiration for GGUF inference
- [shimmytok](https://crates.io/crates/shimmytok) — GGUF tokenizer support
