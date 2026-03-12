# Architecture

This document collects the deeper implementation details that do not belong in the root README.

## Overview

Oxide is split into three main layers:

- CLI for local interaction and model management
- Inference for generation, streaming, batching, and runtime tuning
- Model loading for GGUF metadata, tokenizer handling, and Candle-backed weights

## Core pieces

### CLI

- Argument parsing with `clap`
- Interactive terminal flow with streaming output
- Hugging Face model inspection, download, and local registry management

### Inference

- `Generator` drives prompt formatting, tokenization, prefill, sampling, and decoding
- Streaming emits `PrefillStatus`, `Token`, and `Done` events
- Warmup primes compute paths before the first generation
- Dynamic batching and prefix cache infrastructure are present for lower-latency serving paths

### Model layer

- GGUF metadata is read from the model file
- Tokenizer handling is built around `shimmytok`
- Chat templates are read from GGUF metadata and rendered with `minijinja`
- Candle provides the underlying tensor and transformer runtime

## Inference flow

The high-level flow is:

1. Load GGUF metadata and model weights
2. Read or derive tokenizer data
3. Extract the embedded chat template when available
4. Format messages into a model prompt
5. Tokenize the prompt
6. Run prefill on the prompt tokens
7. Sample one token at a time
8. Decode and stream output tokens

## Chat templates

Oxide expects chat-oriented GGUF models to include `tokenizer.chat_template` metadata.

At runtime it:

1. Reads the template from GGUF metadata
2. Renders it with `minijinja`
3. Feeds the rendered prompt into the tokenizer and generator

If a model does not include a usable chat template, chat workflows may fail or require a different model.

## Supported formats and model families

Oxide targets GGUF model files.

Current support is centered on:

- LLaMA-compatible architectures
- LFM2

Compatible GGUF tokenizer families include:

- SPM
- BPE
- WPM
- UGM
- RWKV

## Performance notes

Oxide is designed for local CPU inference.

Notable implementation choices:

- Memory-mapped model loading
- Token-by-token streaming
- Special-token filtering through tokenizer metadata
- SIMD runtime selection for `avx512`, `avx2`, `neon`, or scalar paths
- Thread count control and thread pinning
- Warmup before first generation
- Tokenizer caching and model download registry support

Some performance-oriented pieces are already present as infrastructure for future work, including dynamic batching and paged cache support.

## Downloaded model storage

- Model files are downloaded into the Hugging Face cache: `~/.cache/huggingface/hub/`
- Oxide keeps a local registry at `~/.oxide/models.json`

## Related docs

- `docs/getting-started.md`
- `docs/api-reference.md`
- `docs/library-usage.md`
- `docs/examples.md`
