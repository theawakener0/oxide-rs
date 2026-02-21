mod model;
mod ui;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use model::ModelLoader;
use ui::App;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Path to tokenizer.json
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Maximum tokens to generate
    #[arg(short, long, default_value = "512")]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// Top-p sampling threshold
    #[arg(long, default_value = "0.9")]
    top_p: f32,

    /// Top-k sampling
    #[arg(long, default_value = "40")]
    top_k: usize,

    /// Repeat penalty
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,

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
                .unwrap_or_else(|_| "oxide=info,wat=error".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    let model_path = if let Some(path) = args.model {
        path
    } else {
        eprintln!("Error: --model is required");
        eprintln!("Usage: oxide --model <path/to/model.gguf>");
        std::process::exit(1);
    };

    let tokenizer_path = args.tokenizer.unwrap_or_else(|| {
        let mut p = model_path.clone();
        p.set_file_name("tokenizer.json");
        p
    });

    tracing::info!("Loading model from: {:?}", model_path);
    tracing::info!("Loading tokenizer from: {:?}", tokenizer_path);

    let model_loader = ModelLoader::new(model_path, tokenizer_path)?;
    let model_info = model_loader.get_model_info();

    tracing::info!(
        "Loaded model: {} ({} params, {} layers)",
        model_info.name,
        model_info.param_count,
        model_info.n_layer
    );

    if args.once {
        let prompt = args
            .prompt
            .unwrap_or_else(|| "Write a hello world program in Rust".to_string());

        let output = model_loader.generate(
            &prompt,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repeat_penalty,
        )?;

        println!("{}", output);
    } else {
        let mut app = App::new(
            model_loader,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repeat_penalty,
        );
        app.run()?;
    }

    Ok(())
}
