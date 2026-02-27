# Library Usage Guide

This guide covers how to use oxide-rs as a library in your Rust projects.

## Adding Oxide-rs to Your Project

```toml
# Cargo.toml
[dependencies]
oxide-rs = "0.1.0"
```

## Core Concepts

### GenerateOptions

Configuration for text generation:

```rust
use oxide_rs::GenerateOptions;

let options = GenerateOptions {
    max_tokens: 512,        // Maximum tokens to generate
    temperature: 0.3,      // Sampling temperature (0.0 = greedy)
    top_p: None,           // Nucleus sampling threshold
    top_k: None,           // Top-k sampling threshold
    repeat_penalty: 1.1,   // Penalty for repeated tokens
    repeat_last_n: 64,     // Context window for repeat penalty
    seed: 299792458,       // Random seed for reproducibility
    system_prompt: None,   // Optional system prompt
};
```

### Model

The `Model` struct provides a builder pattern for more complex usage:

```rust
use oxide_rs::Model;

let mut model = Model::new("model.gguf")?
    .with_options(options)
    .with_tokenizer("tokenizer.json")  // Optional, extracted from GGUF if omitted
    .load()?;
```

## Usage Patterns

### Pattern 1: Simple Function

For quick one-off generations:

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate(
        "model.gguf",
        GenerateOptions::default(),
        "Write a hello world program",
    )?;
    println!("{}", result);
    Ok(())
}
```

### Pattern 2: Builder API

For multiple generations with the same model:

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

    // Multiple generations with same model loaded
    let response1 = model.generate("What is Rust?")?;
    let response2 = model.generate("What is Cargo?")?;

    println!("Rust: {}", response1);
    println!("Cargo: {}", response2);
    Ok(())
}
```

### Pattern 3: Streaming

For real-time output as tokens are generated:

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")
        .load()?;

    model.generate_stream("Tell me a story", |token| {
        print!("{}", token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    })?;

    println!(); // Newline after streaming
    Ok(())
}
```

### Pattern 4: Warmup

Pre-compile compute kernels for faster first-token:

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")
        .load()?;

    // Warmup with 128 tokens (compiles compute kernels)
    model.warmup(128)?;

    // First generation will be faster
    let response = model.generate("Hello!")?;
    Ok(())
}
```

### Pattern 5: Conversation History

Maintain conversation history across generations:

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")
        .with_options(oxide_rs::GenerateOptions {
            system_prompt: Some("You are a helpful assistant.".into()),
            ..Default::default()
        })
        .load()?;

    // First turn
    let response1 = model.generate("What is 2+2?")?;
    println!("User: What is 2+2?");
    println!("Assistant: {}", response1);

    // Second turn - history is maintained automatically
    let response2 = model.generate("Multiply that by 3.")?;
    println!("User: Multiply that by 3.");
    println!("Assistant: {}", response2);

    // Clear history if needed
    model.clear_history();
    Ok(())
}
```

## Accessing Model Metadata

Get information about the loaded model:

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")?.load()?;

    if let Some(metadata) = model.metadata() {
        println!("Model: {}", metadata.name);
        println!("Architecture: {}", metadata.architecture);
        println!("Layers: {}", metadata.n_layer);
        println!("Embedding Size: {}", metadata.n_embd);
        println!("Vocab Size: {}", metadata.vocab_size);
        println!("Context Length: {}", metadata.context_length);
    }
    Ok(())
}
```

## Context Management

Monitor and manage context usage:

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")?.load()?;

    // Get context usage
    let used = model.context_used().unwrap_or(0);
    let limit = model.context_limit().unwrap_or(4096);
    let percentage = model.context_percentage().unwrap_or(0.0);

    println!("Context: {} / {} tokens ({:.1}%)", used, limit, percentage);

    // After generating, check again
    model.generate("Hello!")?;
    let new_used = model.context_used().unwrap_or(0);
    println!("After generation: {} tokens used", new_used);

    Ok(())
}
```

### Streaming vs Non-Streaming

There is **no performance difference** between streaming and non-streaming generation. Both use the same inference engine and generate tokens at the same speed.

- **Non-streaming** (`generate()`): Accumulates all tokens, returns complete response at once
- **Streaming** (`generate_stream()`): Callback fires after each token, enabling real-time display

```rust
// Non-streaming - returns all at once
let response = model.generate("Hello!")?;

// Streaming - callback fires per token
model.generate_stream("Hello!", |token| {
    print!("{}", token);
})?;
```

## Error Handling

All operations return `Result<T, Box<dyn std::error::Error>>`:

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() {
    match generate("model.gguf", GenerateOptions::default(), "Hello") {
        Ok(result) => println!("Success: {}", result),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Best Practices

1. **Reuse Model instances** - Loading a model is expensive; generate multiple times with the same instance
2. **Use warmup** - Call `warmup()` after loading for faster first-token generation
3. **Adjust max_tokens** - Set appropriate limits based on your use case
4. **Handle streaming for long outputs** - Provides better UX for generated text
5. **Set appropriate repeat_penalty** - Higher values (1.2+) reduce repetition but may affect coherence
6. **Monitor context** - Use `context_used()` and `context_percentage()` to avoid exceeding limits
7. **Clear history when needed** - Call `clear_history()` to reset context for new conversations
