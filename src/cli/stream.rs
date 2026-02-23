use std::io::{self, Write};
use std::time::Instant;

use crossterm::{
    execute,
    style::{Attribute, Print, ResetColor, SetAttribute, SetForegroundColor},
};

use super::theme::Theme;

pub struct StreamOutput {
    stdout: io::Stdout,
    first_token: bool,
    start_time: Instant,
    token_count: usize,
}

impl StreamOutput {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
            first_token: true,
            start_time: Instant::now(),
            token_count: 0,
        }
    }

    pub fn print_token(&mut self, token: &str) {
        if self.first_token {
            execute!(
                self.stdout,
                SetForegroundColor(Theme::ACCENT_CYAN),
                SetAttribute(Attribute::Bold),
                Print("◆ "),
                ResetColor,
                SetForegroundColor(Theme::TEXT_PRIMARY),
            )
            .ok();
            self.first_token = false;
        }

        self.token_count += 1;

        execute!(self.stdout, Print(token)).ok();
        self.stdout.flush().ok();
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
                "{} tokens • {:.1} tok/s • {:.1}s",
                self.token_count,
                tokens_per_sec,
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
