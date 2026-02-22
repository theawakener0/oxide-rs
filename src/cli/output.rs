use std::io::{self, Write};
use std::time::Duration;

use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
};

pub struct Output {
    stdout: io::Stdout,
}

impl Output {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
        }
    }

    pub fn print_model_info(&mut self, name: &str, params: &str, quant: &str) {
        execute!(
            self.stdout,
            SetForegroundColor(Color::DarkGrey),
            Print("  Model: "),
            ResetColor,
            Print(format!("{}\n", name)),
            SetForegroundColor(Color::DarkGrey),
            Print("  Size:  "),
            ResetColor,
            Print(format!("{}\n", params)),
            SetForegroundColor(Color::DarkGrey),
            Print("  Quant: "),
            ResetColor,
            Print(format!("{}\n", quant)),
        )
        .ok();
    }

    pub fn print_separator(&mut self) {
        execute!(
            self.stdout,
            SetForegroundColor(Color::DarkGrey),
            Print("\n"),
            Print("━".repeat(60)),
            Print("\n"),
            ResetColor,
        )
        .ok();
    }

    pub fn print_prompt(&mut self, text: &str) {
        execute!(
            self.stdout,
            SetForegroundColor(Color::Green),
            Print("You: "),
            ResetColor,
            Print(format!("{}\n", text)),
        )
        .ok();
        self.print_separator();
    }

    pub fn print_assistant_prefix(&mut self) {
        execute!(
            self.stdout,
            SetForegroundColor(Color::Cyan),
            Print("Assistant: "),
            ResetColor,
        )
        .ok();
    }

    pub fn print_stats(&mut self, tokens: usize, duration: Duration) {
        let tps = if duration.as_secs_f64() > 0.0 {
            tokens as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        execute!(
            self.stdout,
            Print("\n"),
            SetForegroundColor(Color::DarkGrey),
            Print(format!(
                "  [Generated {} tokens • {:.1} tok/s]",
                tokens, tps
            )),
            ResetColor,
            Print("\n"),
        )
        .ok();
    }

    pub fn print_cancelled(&mut self) {
        execute!(
            self.stdout,
            Print("\n"),
            SetForegroundColor(Color::Yellow),
            Print("  [Generation cancelled]"),
            ResetColor,
            Print("\n"),
        )
        .ok();
    }

    pub fn print_error(&mut self, msg: &str) {
        execute!(
            self.stdout,
            SetForegroundColor(Color::Red),
            Print(format!("Error: {}\n", msg)),
            ResetColor,
        )
        .ok();
    }

    pub fn print_welcome(&mut self) {
        execute!(
            self.stdout,
            SetForegroundColor(Color::DarkGrey),
            Print("  Type your message and press Enter. Press ESC to cancel, Ctrl+C to exit.\n"),
            ResetColor,
        )
        .ok();
    }

    pub fn print_input_prompt(&mut self) {
        execute!(
            self.stdout,
            SetForegroundColor(Color::Green),
            Print("\nYou: "),
            ResetColor,
        )
        .ok();
    }
}

pub struct StreamOutput {
    stdout: io::Stdout,
    first_token: bool,
}

impl StreamOutput {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
            first_token: true,
        }
    }

    pub fn print_token(&mut self, token: &str) {
        if self.first_token {
            execute!(
                self.stdout,
                SetForegroundColor(Color::Cyan),
                Print("Assistant: "),
                ResetColor,
            )
            .ok();
            self.first_token = false;
        }
        execute!(self.stdout, Print(token)).ok();
        self.stdout.flush().ok();
    }

    pub fn finish(&mut self) {
        execute!(self.stdout, Print("\n")).ok();
    }

    pub fn reset(&mut self) {
        self.first_token = true;
    }
}

impl Default for Output {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for StreamOutput {
    fn default() -> Self {
        Self::new()
    }
}
