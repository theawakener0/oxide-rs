use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use oxide_rs::cli::download::DownloadProgressBar;
use oxide_rs::cli::{
    print_banner, print_divider, print_model_info, print_welcome, ModelLoader, PromptDisplay,
    Spinner, StreamOutput, ThinkingSpinner,
};
use oxide_rs::inference::{
    init_simd, init_thread_pinner, simd_dispatch::SimdLevel, thread_pinner::ThreadPinnerConfig,
    Generator, StreamEvent,
};
use oxide_rs::model::download::find_gguf_file;
use oxide_rs::model::{
    download_model, format_size, get_model_info, list_models, register_model, unregister_model,
};
use oxide_rs::server::run as server_run;
use oxide_rs::tui::state::Screen;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Download a model from HuggingFace Hub
    #[arg(short, long)]
    download: Option<String>,

    /// List all locally downloaded models
    #[arg(short, long)]
    models: bool,

    /// Show information about a model on HuggingFace Hub
    #[arg(long)]
    info: Option<String>,

    /// Remove a model from local storage
    #[arg(long)]
    remove: Option<String>,

    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Path to tokenizer.json (optional, will extract from GGUF if not provided)
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Maximum tokens to generate
    #[arg(short, long, default_value = "512")]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(long, default_value = "0.3")]
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

    /// Batch size for warmup/prefill (default: 128)
    #[arg(long, default_value = "128")]
    batch_size: usize,

    /// Random seed
    #[arg(long, default_value = "299792458")]
    seed: u64,

    /// Number of threads for inference (default: auto-detect)
    #[arg(long)]
    threads: Option<usize>,

    /// System prompt for the model
    #[arg(short, long)]
    system: Option<String>,

    /// Prompt to use (if not using interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Run in non-interactive mode (generate and exit)
    #[arg(short, long)]
    once: bool,

    /// Maximum batch size for dynamic batching (default: 8)
    #[arg(long, default_value = "8")]
    max_batch_size: usize,

    /// Batch window in milliseconds (default: 100ms)
    #[arg(long, default_value = "100")]
    batch_window_ms: u64,

    /// SIMD level (auto/avx512/avx2/neon/scalar)
    #[arg(long, default_value = "auto")]
    simd: String,

    /// Launch TUI mode instead of CLI chat
    #[arg(short, long)]
    tui: bool,

    /// Run as OpenAI-compatible HTTP server
    #[arg(long)]
    server: bool,

    /// Port for HTTP server (default: 8080)
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Host for HTTP server (default: 0.0.0.0)
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(ref repo_id) = cli.download {
        handle_download(repo_id)?;
        return Ok(());
    }

    if cli.models {
        handle_list_models()?;
        return Ok(());
    }

    if let Some(ref repo_id) = cli.info {
        handle_info(repo_id)?;
        return Ok(());
    }

    if let Some(ref model_id) = cli.remove {
        handle_remove(model_id)?;
        return Ok(());
    }

    if cli.tui {
        let initial_screen = if cli.models {
            Some(Screen::Models)
        } else {
            None
        };
        oxide_rs::tui::run(cli.model.clone(), cli.download.clone(), initial_screen)?;
        return Ok(());
    }

    if cli.server {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "oxide_rs=info".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();

        let runtime = tokio::runtime::Runtime::new()?;
        if let Err(e) = runtime.block_on(server_run(cli.host, cli.port)) {
            eprintln!("Server error: {}", e);
        }
        return Ok(());
    }

    let model_path = cli.model.clone().ok_or_else(|| {
        anyhow::anyhow!("No model specified. Use --model or --download to get a model.")
    })?;

    run_inference(cli, model_path)
}

fn handle_download(repo_id: &str) -> Result<()> {
    println!();
    print_banner();
    print_divider();
    println!();

    let spinner = Spinner::new("Fetching model info...");
    spinner.finish();

    let info = get_model_info(repo_id)?;

    let gguf_file = find_gguf_file(&info.files).ok_or_else(|| {
        anyhow::anyhow!(
            "No GGUF file found in repository. This model may not have GGUF files available."
        )
    })?;

    println!();
    println!("  Repository: {}", repo_id);
    println!("  File:       {}", gguf_file.rfilename);
    println!("  Size:       {}", format_size(gguf_file.size));
    println!();
    print_divider();
    println!();

    let mut progress_bar = DownloadProgressBar::new(&gguf_file.rfilename, gguf_file.size);

    let (path, filename): (std::path::PathBuf, String) =
        download_model(repo_id, Some(&gguf_file.rfilename), |progress| {
            progress_bar.update(&progress);
        })?;

    progress_bar.finish(path.to_str().unwrap_or(""));

    let entry = register_model(repo_id, &filename, path.clone(), gguf_file.size)?;

    println!("  Model ID:   {}", entry.id);
    println!();
    println!("  Run with:    oxide-rs --model {}", path.display());

    Ok(())
}

fn handle_list_models() -> Result<()> {
    let models = list_models()?;

    println!();
    print_banner();
    println!();

    if models.is_empty() {
        println!("  No models downloaded yet.");
        println!();
        println!("  Download a model with:");
        println!("    oxide-rs --download <repo-id>");
        println!();
        return Ok(());
    }

    println!("  📦 Local Models");
    print_divider();

    for model in &models {
        let size_str = format_size(model.size_bytes);
        let quant = model.quantization.as_deref().unwrap_or("Unknown");
        println!("  {}", model.id);
        println!("    Size:     {}", size_str);
        println!("    Quant:    {}", quant);
        println!("    Path:     {}", model.path.display());
        println!();
    }

    println!("  Run a model:");
    println!("    oxide-rs --model <path>");
    println!();

    Ok(())
}

fn handle_info(repo_id: &str) -> Result<()> {
    println!();
    println!("  Fetching model info...");

    let info = get_model_info(repo_id)?;

    println!();
    print_banner();
    println!();

    println!("  Repository: {}", info.repo_id);
    println!("  Files:      {}", info.files.len());
    println!("  Total Size: {}", format_size(info.total_size));
    println!();

    let gguf_files: Vec<_> = info
        .files
        .iter()
        .filter(|f| f.rfilename.ends_with(".gguf"))
        .collect();

    if !gguf_files.is_empty() {
        println!("  GGUF Files:");
        for file in &gguf_files {
            println!("    • {} ({})", file.rfilename, format_size(file.size));
        }
        println!();

        if let Some(best) = find_gguf_file(&info.files) {
            println!(
                "  Recommended: {} ({})",
                best.rfilename,
                format_size(best.size)
            );
        }
    } else {
        println!("  All Files:");
        for file in &info.files {
            println!("    • {} ({})", file.rfilename, format_size(file.size));
        }
    }

    println!();
    println!("  Download with:");
    println!("    oxide-rs --download {}", repo_id);
    println!();

    Ok(())
}

fn handle_remove(model_id: &str) -> Result<()> {
    if let Some(entry) = unregister_model(model_id)? {
        println!();
        println!("  ✓ Removed model: {}", entry.id);
        println!("    File: {}", entry.path.display());
        println!();
    } else {
        println!();
        println!("  Model not found: {}", model_id);
        println!();
        println!("  Use 'oxide-rs --models' to see available models.");
        println!();
    }

    Ok(())
}

fn run_inference(cli: Cli, model_path: PathBuf) -> Result<()> {
    let num_cpus = num_cpus::get();
    let num_threads = cli
        .threads
        .unwrap_or_else(|| num_cpus.saturating_sub(1).max(1));

    let simd_level = SimdLevel::from_str(&cli.simd);
    let simd = init_simd(simd_level);
    tracing::info!(
        "SIMD: {:?} (AVX512: {}, AVX2: {}, NEON: {})",
        simd.level,
        simd.cpu_features.has_avx512,
        simd.cpu_features.has_avx2,
        simd.cpu_features.has_neon
    );

    unsafe { std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string()) };
    tracing::info!(
        "Using {} threads for inference (RAYON_NUM_THREADS set)",
        num_threads
    );

    let tokenizer_path = cli.tokenizer.clone();
    let (temperature, top_p, top_k, seed, batch_size) = (
        cli.temperature,
        cli.top_p,
        cli.top_k,
        cli.seed,
        cli.batch_size,
    );
    let system_prompt = cli.system.clone();

    let load_handle = std::thread::spawn(move || {
        Generator::new(
            &model_path,
            tokenizer_path.as_ref(),
            temperature,
            top_p,
            top_k,
            seed,
            system_prompt,
            batch_size,
        )
    });

    let thread_pinner = init_thread_pinner(ThreadPinnerConfig::auto(num_cpus));
    tracing::info!(
        "Thread pinning: {} threads on cores {:?}",
        thread_pinner.num_threads(),
        thread_pinner.core_ids()
    );

    let pinned_pool = thread_pinner
        .build_thread_pool()
        .map_err(|e| anyhow::anyhow!("Failed to build pinned thread pool: {}", e))?;
    pinned_pool.install(|| {
        tracing::info!(
            "Thread pool initialized with {} pinned threads",
            num_threads
        );
    });

    print_banner();

    let loader = ModelLoader::new();

    let mut generator = match load_handle.join() {
        Ok(Ok(g)) => g,
        Ok(Err(e)) => {
            loader.finish_with_error(&format!("Failed: {}", e));
            return Err(e);
        }
        Err(_) => {
            loader.finish_with_error("Model loading thread panicked");
            return Err(anyhow::anyhow!("Model loading thread panicked"));
        }
    };

    if let Err(e) = pinned_pool.install(|| generator.warmup(1)) {
        tracing::warn!("Model warmup failed: {}", e);
    }

    let metadata = generator.metadata().clone();
    loader.finish(&metadata.name);

    print_model_info(
        &metadata.name,
        &format_size(metadata.file_size),
        metadata.quantization.as_deref().unwrap_or("Unknown"),
        metadata.n_layer,
        metadata.n_embd,
        metadata.context_length,
    );

    if cli.once {
        let prompt = cli
            .prompt
            .unwrap_or_else(|| "Write a hello world program in Rust".to_string());

        let mut prompt_display = PromptDisplay::new();
        prompt_display.show_user_input(&prompt);

        let mut gen_output = generator;
        let mut stream = StreamOutput::new();
        let mut thinking_spinner: Option<ThinkingSpinner> = None;
        let context_limit = gen_output.context_limit();
        let context_used = gen_output.context_used();
        let mut prompt_token_count = 0usize;

        pinned_pool.install(|| {
            gen_output.generate_streaming(
                &prompt,
                cli.max_tokens,
                cli.repeat_penalty,
                cli.repeat_last_n,
                |event| match event {
                    StreamEvent::PrefillStatus(count) => {
                        prompt_token_count = count;
                        stream.set_prompt_tokens(count);
                        if thinking_spinner.is_none() {
                            thinking_spinner = Some(ThinkingSpinner::new());
                        }
                    }
                    StreamEvent::Token(t) => {
                        if let Some(spinner) = thinking_spinner.take() {
                            spinner.stop();
                        }
                        stream.set_context(context_used, context_limit);
                        stream.print_token(&t);
                    }
                    StreamEvent::Done => {
                        stream.finish();
                    }
                },
            )
        })?;

        return Ok(());
    }

    print_divider();
    print_welcome();
    print_divider();

    interactive_mode(generator, cli, pinned_pool)
}

fn interactive_mode(generator: Generator, cli: Cli, pinned_pool: rayon::ThreadPool) -> Result<()> {
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
            generator.clear_history();
            println!("  History cleared.\n");
            continue;
        }

        if prompt == "/help" {
            println!("  Commands:");
            println!("    /clear   - Clear conversation history");
            println!("    /context - Show context usage");
            println!("    /stats   - Show model info and settings");
            println!("    /exit    - Exit the program");
            println!("    /help    - Show this help\n");
            continue;
        }

        if prompt == "/context" {
            let used = generator.context_used();
            let limit = generator.context_limit();
            let percentage = generator.context_percentage();
            println!(
                "  Context: {} / {} tokens ({:.1}%)\n",
                format_token_count(used),
                format_token_count(limit),
                percentage
            );
            continue;
        }

        if prompt == "/stats" {
            let meta = generator.metadata();
            println!("  Model:     {}", meta.name);
            println!(
                "  Quant:     {}",
                meta.quantization.as_deref().unwrap_or("Unknown")
            );
            println!(
                "  Context:   {} tokens",
                format_token_count(meta.context_length)
            );
            println!("  Layers:    {}", meta.n_layer);
            println!("  Embedding: {}", meta.n_embd);
            println!("  Vocab:     {}", meta.vocab_size);
            println!("  Temp:      {}", cli.temperature);
            println!("  Max Tok:   {}", cli.max_tokens);
            println!("  Seed:      {}", cli.seed);
            println!();
            continue;
        }

        let mut stream = StreamOutput::new();
        let mut thinking_spinner: Option<ThinkingSpinner> = None;
        let context_limit = generator.context_limit();
        let context_used = generator.context_used();
        let mut prompt_token_count = 0usize;

        pinned_pool.install(|| {
            generator.generate_streaming(
                &prompt,
                cli.max_tokens,
                cli.repeat_penalty,
                cli.repeat_last_n,
                |event| match event {
                    StreamEvent::PrefillStatus(count) => {
                        prompt_token_count = count;
                        stream.set_prompt_tokens(count);
                        if thinking_spinner.is_none() {
                            thinking_spinner = Some(ThinkingSpinner::new());
                        }
                    }
                    StreamEvent::Token(t) => {
                        if let Some(spinner) = thinking_spinner.take() {
                            spinner.stop();
                        }
                        stream.set_context(context_used, context_limit);
                        stream.print_token(&t);
                    }
                    StreamEvent::Done => {
                        stream.finish();
                    }
                },
            )
        })?;

        print_divider();
    }

    Ok(())
}

fn format_token_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
