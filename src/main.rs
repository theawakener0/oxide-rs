mod cli;
mod inference;
mod model;

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use cli::{
    print_banner, print_divider, print_model_info, print_welcome, History, ModelLoader,
    PromptDisplay, StreamOutput,
};
use inference::{Generator, StreamEvent};

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

    /// Clear conversation history
    #[arg(long)]
    clear_history: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "oxide=error".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    print_banner();

    let loader = ModelLoader::new();

    let generator = match Generator::new(
        &args.model,
        args.tokenizer.as_ref(),
        args.temperature,
        args.top_p,
        args.top_k,
        args.seed,
        args.max_history,
    ) {
        Ok(g) => g,
        Err(e) => {
            loader.finish_with_error(&format!("Failed: {}", e));
            return Err(e);
        }
    };

    let metadata = generator.metadata().clone();
    loader.finish(&metadata.name);

    print_model_info(
        &metadata.name,
        &format_size(metadata.file_size),
        "Q4_K_M",
        metadata.n_layer,
        metadata.n_embd,
    );

    if args.clear_history {
        let mut history = History::load();
        history.clear();
        println!("  History cleared.");
    }

    if args.once {
        let prompt = args
            .prompt
            .unwrap_or_else(|| "Write a hello world program in Rust".to_string());

        let mut prompt_display = PromptDisplay::new();
        prompt_display.show_user_input(&prompt);

        let mut gen = generator;
        let mut stream = StreamOutput::new();

        gen.generate(
            &prompt,
            args.max_tokens,
            args.repeat_penalty,
            args.repeat_last_n,
            |event| match event {
                StreamEvent::Token(t) => {
                    stream.print_token(&t);
                }
                StreamEvent::Done => {
                    stream.finish();
                }
            },
        )?;

        return Ok(());
    }

    print_divider();
    print_welcome();
    print_divider();

    interactive_mode(generator, args)
}

fn interactive_mode(generator: Generator, args: Args) -> Result<()> {
    let mut history = History::load();
    let mut generator = generator;
    let mut prompt_display = PromptDisplay::new();

    loop {
        prompt_display.show_input_prompt();
        io::stdout().flush()?;

        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt)?;
        let prompt = prompt.trim().to_string();

        if prompt.is_empty() {
            continue;
        }

        if prompt == "/exit" || prompt == "/quit" {
            break;
        }

        if prompt == "/clear" {
            history.clear();
            println!("  History cleared.\n");
            continue;
        }

        if prompt == "/help" {
            println!("  Commands:");
            println!("    /clear  - Clear conversation history");
            println!("    /exit   - Exit the program");
            println!("    /help   - Show this help\n");
            continue;
        }

        let mut stream = StreamOutput::new();

        history.add("user", &prompt);

        generator.generate(
            &prompt,
            args.max_tokens,
            args.repeat_penalty,
            args.repeat_last_n,
            |event| match event {
                StreamEvent::Token(t) => {
                    stream.print_token(&t);
                }
                StreamEvent::Done => {
                    stream.finish();
                }
            },
        )?;

        history.save();
        print_divider();
    }

    history.save();
    Ok(())
}

fn format_size(size: u64) -> String {
    if size < 1_000 {
        format!("{}B", size)
    } else if size < 1_000_000 {
        format!("{:.1}KB", size as f64 / 1e3)
    } else if size < 1_000_000_000 {
        format!("{:.1}MB", size as f64 / 1e6)
    } else {
        format!("{:.1}GB", size as f64 / 1e9)
    }
}
