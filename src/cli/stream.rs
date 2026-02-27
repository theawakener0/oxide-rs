use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::{
    cursor::MoveToColumn,
    execute,
    style::{Attribute, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{Clear, ClearType},
};

use super::theme::Theme;

const THINKING_FRAMES: &[&str] = &[
    "🦀💭 Thinking.",
    "🦀💭 Thinking..",
    "🦀💭 Thinking...",
    "🦀💭 Thinking",
];

pub struct ThinkingSpinner {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ThinkingSpinner {
    pub fn new() -> Self {
        let running = Arc::new(AtomicBool::new(true));

        let handle = thread::spawn({
            let running = running.clone();
            move || {
                let mut stdout = io::stdout();
                let mut i = 0usize;

                while running.load(Ordering::Relaxed) {
                    let frame = THINKING_FRAMES[i % THINKING_FRAMES.len()];

                    execute!(
                        stdout,
                        MoveToColumn(0),
                        Clear(ClearType::CurrentLine),
                        SetForegroundColor(Theme::ACCENT_CYAN),
                        Print(frame),
                        ResetColor
                    )
                    .ok();

                    stdout.flush().ok();
                    thread::sleep(Duration::from_millis(200));
                    i = i.wrapping_add(1);
                }
            }
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    pub fn stop(mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }

        let mut stdout = io::stdout();
        execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine)).ok();
    }
}

impl Default for ThinkingSpinner {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ThinkingSpinner {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

pub fn strip_special_tokens(input: &str) -> String {
    let mut result = input.to_string();

    let patterns = [
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_start|>system",
        "<|im_start|>",
        "<|im_end|>",
        "<|end|>",
        "<|start|>",
        "<|sep|>",
        "<s>",
        "</s>",
        "<pad>",
    ];

    for pattern in &patterns {
        result = result.replace(pattern, "");
    }

    result = result.replace("<br>", "\n");
    result = result.replace("\\n", "\n");

    while result.contains("\n\n\n") {
        result = result.replace("\n\n\n", "\n\n");
    }

    result.trim().to_string()
}

pub struct StreamOutput {
    stdout: io::Stdout,
    first_token: bool,
    start_time: Instant,
    token_count: usize,
    last_stats_time: Instant,
    context_used: usize,
    context_limit: usize,
}

impl StreamOutput {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
            first_token: true,
            start_time: Instant::now(),
            token_count: 0,
            last_stats_time: Instant::now(),
            context_used: 0,
            context_limit: 4096,
        }
    }

    pub fn set_context(&mut self, used: usize, limit: usize) {
        self.context_used = used;
        self.context_limit = limit;
    }

    pub fn print_token(&mut self, token: &str) {
        if self.first_token {
            self.first_token = false;
        }

        self.token_count += 1;

        let cleaned = strip_special_tokens(token);

        if cleaned.contains('\n') {
            for line in cleaned.lines() {
                execute!(self.stdout, Print(line)).ok();
                execute!(self.stdout, Print("\n")).ok();
            }
        } else {
            execute!(self.stdout, Print(&cleaned)).ok();
        }

        self.stdout.flush().ok();

        if self.last_stats_time.elapsed() >= Duration::from_millis(500) {
            self.print_live_stats();
            self.last_stats_time = Instant::now();
        }
    }

    fn print_live_stats(&mut self) {
        let elapsed = self.start_time.elapsed();
        let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
            self.token_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        execute!(
            self.stdout,
            SetForegroundColor(Theme::IRON_GRAY),
            Print("\n  ▸ "),
            ResetColor,
            SetForegroundColor(Theme::TEXT_SECONDARY),
            Print(format!(
                "{} tokens • {:.1} tok/s • Context: {}/{}",
                self.token_count, tokens_per_sec, self.context_used, self.context_limit
            )),
            ResetColor,
            Print("\n")
        )
        .ok();
    }

    pub fn finish(&mut self) {
        let elapsed = self.start_time.elapsed();
        let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
            self.token_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        execute!(
            self.stdout,
            ResetColor,
            Print("\n"),
            SetForegroundColor(Theme::IRON_GRAY),
            Print("  ▸ "),
            ResetColor,
            SetForegroundColor(Theme::TEXT_SECONDARY),
            Print(format!(
                "{} tokens • {:.1} tok/s • Context: {}/{} • {:.1}s",
                self.token_count,
                tokens_per_sec,
                self.context_used,
                self.context_limit,
                elapsed.as_secs_f64()
            )),
            ResetColor,
            Print("\n")
        )
        .ok();
    }
}

impl Default for StreamOutput {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PromptDisplay {
    stdout: io::Stdout,
}

impl PromptDisplay {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
        }
    }

    pub fn show_user_input(&mut self, text: &str) {
        execute!(
            self.stdout,
            Print("\n"),
            SetForegroundColor(Theme::RUST_ORANGE),
            SetAttribute(Attribute::Bold),
            Print("▸ "),
            ResetColor,
            SetForegroundColor(Theme::TEXT_PRIMARY),
            Print(text),
            ResetColor,
            Print("\n\n")
        )
        .ok();
        self.stdout.flush().ok();
    }

    pub fn show_input_prompt(&mut self) {
        execute!(
            self.stdout,
            Print("\n"),
            SetForegroundColor(Theme::RUST_ORANGE),
            SetAttribute(Attribute::Bold),
            Print("▸"),
            ResetColor,
            Print(" ")
        )
        .ok();
        self.stdout.flush().ok();
    }
}

impl Default for PromptDisplay {
    fn default() -> Self {
        Self::new()
    }
}

pub fn print_welcome() {
    let mut stdout = io::stdout();
    execute!(
        stdout,
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  Type your message and press Enter. "),
        SetForegroundColor(Theme::TEXT_SECONDARY),
        Print("Ctrl+C"),
        ResetColor,
        SetForegroundColor(Theme::IRON_GRAY),
        Print(" to exit.\n"),
        ResetColor
    )
    .ok();
}
