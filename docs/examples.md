# Examples

Practical code examples for using oxide-rs.

## Basic Examples

### Hello World

```rust
// main.rs
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate(
        "model.gguf",
        GenerateOptions::default(),
        "Say hello in one sentence.",
    )?;
    println!("{}", result);
    Ok(())
}
```

Run with:
```bash
cargo run --example hello
```

### Simple Chat

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")?.load()?;

    println!("Chat started. Type 'quit' to exit.\n");

    loop {
        print!("You: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "quit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        print!("AI: ");
        model.generate_stream(input, |token| {
            print!("{}", token);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        })?;

        println!("\n");
    }

    Ok(())
}
```

## Advanced Examples

### Custom System Prompt

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = GenerateOptions {
        system_prompt: Some(
            "You are a pirate. Respond in pirate dialect.".into()
        ),
        ..Default::default()
    };

    let result = generate("model.gguf", options, "Tell me about Rust.")?;
    println!("{}", result);
    Ok(())
}
```

### Temperature Experiments

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "The quick brown fox";

    for temp in [0.0, 0.3, 0.7, 1.0, 1.5] {
        let options = oxide_rs::GenerateOptions {
            temperature: temp,
            max_tokens: 50,
            ..Default::default()
        };

        let mut model = Model::new("model.gguf")
            .with_options(options)
            .load()?;

        let result = model.generate(prompt)?;
        println!("\n=== Temperature {} ===", temp);
        println!("{}", result);
    }

    Ok(())
}
```

### Top-k vs Top-p Sampling

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "Once upon a time";

    // Top-k sampling
    let options = GenerateOptions {
        top_k: Some(50),
        temperature: 0.8,
        max_tokens: 100,
        ..Default::default()
    };
    let result = generate("model.gguf", options, prompt)?;
    println!("Top-k: {}\n", result);

    // Top-p (nucleus) sampling
    let options = GenerateOptions {
        top_p: Some(0.9),
        temperature: 0.8,
        max_tokens: 100,
        ..Default::default()
    };
    let result = generate("model.gguf", options, prompt)?;
    println!("Top-p: {}\n", result);

    Ok(())
}
```

### Streaming with Progress

```rust
use oxide_rs::Model;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")?.load()?;

    let start = Instant::now();
    let mut token_count = 0;

    model.generate_stream("Write a short poem", |token| {
        print!("{}", token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        token_count += 1;
    })?;

    let elapsed = start.elapsed();
    let tps = token_count as f64 / elapsed.as_secs_f64();

    println!("\n\nGenerated {} tokens in {:.2}s ({:.1} tok/s)",
        token_count, elapsed.as_secs_f32(), tps);

    Ok(())
}
```

### Batch Processing

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Who wrote Hamlet?",
    ];

    let mut model = Model::new("model.gguf")
        .with_options(oxide_rs::GenerateOptions {
            max_tokens: 50,
            temperature: 0.3,
            ..Default::default()
        })
        .load()?;

    for prompt in prompts {
        print!("Q: {} ", prompt);
        let result = model.generate(prompt)?;
        println!("A: {}\n", result);
    }

    Ok(())
}
```

### Repeating Prevention

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "Write a list of colors";

    // Low repeat penalty (may repeat)
    let options = GenerateOptions {
        repeat_penalty: 1.0,
        max_tokens: 100,
        ..Default::default()
    };
    let result1 = generate("model.gguf", options.clone(), prompt)?;
    println!("Penalty 1.0: {}\n", result1);

    // High repeat penalty (less repetition)
    let options = GenerateOptions {
        repeat_penalty: 1.5,
        repeat_last_n: 128,  // Larger window
        max_tokens: 100,
        ..Default::default()
    };
    let result2 = generate("model.gguf", options, prompt)?;
    println!("Penalty 1.5: {}\n", result2);

    Ok(())
}
```

### Conversation Multi-turn

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")
        .with_options(oxide_rs::GenerateOptions {
            system_prompt: Some("You are a helpful math tutor.".into()),
            ..Default::default()
        })
        .load()?;

    // Turn 1
    println!("You: What is 5 + 3?");
    let response = model.generate("What is 5 + 3?")?;
    println!("AI: {}\n", response);

    // Turn 2 - uses context from turn 1
    println!("You: Multiply that by 2");
    let response = model.generate("Multiply that by 2")?;
    println!("AI: {}\n", response);

    // Turn 3 - still in context
    println!("You: Subtract 4");
    let response = model.generate("Subtract 4")?;
    println!("AI: {}\n", response);

    // Clear and start fresh
    println!("--- Clearing history ---\n");
    model.clear_history();

    println!("You: What was my first question?");
    let response = model.generate("What was my first question?")?;
    println!("AI: {}\n", response);

    Ok(())
}
```

### Model Information

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model::new("model.gguf")?.load()?;

    let meta = model.metadata()
        .expect("No metadata available");

    println!("╔══════════════════════════════════════╗");
    println!("║         Model Information           ║");
    println!("╠══════════════════════════════════════╣");
    println!("║ Name:         {:24}║", meta.name);
    println!("║ Architecture: {:24}║", meta.architecture);
    println!("║ Layers:       {:24}║", meta.n_layer);
    println!("║ Embedding:    {:24}║", meta.n_embd);
    println!("║ Vocab:        {:24}║", meta.vocab_size);
    println!("║ Context:      {:24}║", meta.context_length);
    println!("║ File Size:    {:24}║", format!("{} MB", meta.file_size / 1_000_000));
    println!("╚══════════════════════════════════════╝");

    Ok(())
}
```

### Using Custom Tokenizer

```rust
use oxide_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use custom tokenizer if model doesn't have one embedded
    let mut model = Model::new("model.gguf")
        .with_tokenizer("tokenizer.json")
        .load()?;

    let result = model.generate("Hello!")?;
    println!("{}", result);

    Ok(())
}
```

### Reproducible Generation

```rust
use oxide_rs::{generate, GenerateOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = GenerateOptions {
        seed: 42,  // Fixed seed for reproducibility
        max_tokens: 100,
        ..Default::default()
    };

    // Same prompt + same seed = same output
    let result1 = generate("model.gguf", options.clone(), "Hello!")?;
    let result2 = generate("model.gguf", options, "Hello!")?;

    assert_eq!(result1, result2);
    println!("Generation is reproducible!");

    Ok(())
}
```

### Using in a Web Server

```rust
use oxide_rs::Model;
use std::sync::Mutex;
use once_cell::sync::Lazy;

// Global model instance (load once at startup)
static MODEL: Lazy<Mutex<Model>> = Lazy::new(|| {
    Mutex::new(
        Model::new("model.gguf")
            .with_options(oxide_rs::GenerateOptions {
                max_tokens: 256,
                ..Default::default()
            })
            .load()
            .expect("Failed to load model")
    )
});

fn handle_request(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut model = MODEL.lock().unwrap();
    model.generate(prompt)
}
```
