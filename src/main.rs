mod inference;
mod model;
mod ui;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use inference::{Generator, StreamEvent};
use ui::App;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to tokenizer.json (optional, will extract from GGUF if not provided)
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Maximum tokens to generate
    #[arg(short, long, default_value = "512")]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(long, default_value = "0.7")]
    temperature: f64,

    /// Top-p sampling threshold
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling
    #[arg(long)]
    top_k: Option<usize>,

    /// Repeat penalty
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,

    /// Context size for repeat penalty
    #[arg(long, default_value = "64")]
    repeat_last_n: usize,

    /// Random seed
    #[arg(long, default_value = "299792458")]
    seed: u64,

    /// Maximum conversation history in tokens
    #[arg(long, default_value = "2048")]
    max_history: usize,

    /// Prompt to use (if not using interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Run in non-interactive mode (generate and exit)
    #[arg(short, long)]
    once: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "oxide=info,candle=error".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let generator = Generator::new(
        &args.model,
        args.tokenizer.as_ref(),
        args.temperature,
        args.top_p,
        args.top_k,
        args.seed,
        args.max_history,
    )?;

    let metadata = generator.metadata().clone();

    tracing::info!(
        "Model: {} ({} layers, {} dim, {} context, {} vocab)",
        metadata.name,
        metadata.n_layer,
        metadata.n_embd,
        metadata.context_length,
        metadata.vocab_size
    );

    if args.once {
        let prompt = args
            .prompt
            .unwrap_or_else(|| "Write a hello world program in Rust".to_string());

        println!("Prompt: {}\n", prompt);
        println!("---");

        let mut gen = generator;
        gen.generate(
            &prompt,
            args.max_tokens,
            args.repeat_penalty,
            args.repeat_last_n,
            |event| match event {
                StreamEvent::Token(t) => print!("{}", t),
                StreamEvent::Done { tokens_generated } => {
                    println!("\n---\nGenerated {} tokens", tokens_generated);
                }
                StreamEvent::Error(e) => eprintln!("Error: {}", e),
            },
        )?;

        println!();
    } else {
        let mut app = App::new(
            generator,
            args.model,
            args.tokenizer,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repeat_penalty,
            args.repeat_last_n,
            args.seed,
            args.max_history,
        );
        app.run()?;
    }

    Ok(())
}
