use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::{
    cursor::MoveToColumn,
    execute,
    style::{Attribute, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::Clear,
    terminal::ClearType,
};

use super::theme::Theme;

const FERRIS_WALKING: &[&str] = &[
    "🦀      ",
    " 🦀     ",
    "  🦀    ",
    "   🦀   ",
    "    🦀  ",
    "     🦀 ",
    "      🦀",
    "     🦀 ",
    "  🦀    ",
    " 🦀     ",
];

pub struct ModelLoader {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ModelLoader {
    pub fn new() -> Self {
        let running = Arc::new(AtomicBool::new(true));

        let handle = thread::spawn({
            let running = running.clone();
            move || {
                let mut stdout = io::stdout();
                let mut i = 0usize;

                while running.load(Ordering::Relaxed) {
                    let ferris = FERRIS_WALKING[i % FERRIS_WALKING.len()];

                    execute!(
                        stdout,
                        MoveToColumn(0),
                        Clear(ClearType::CurrentLine),
                        SetForegroundColor(Theme::RUST_ORANGE),
                        Print(ferris),
                        ResetColor
                    )
                    .ok();

                    stdout.flush().ok();
                    thread::sleep(Duration::from_millis(100));
                    i = i.wrapping_add(1);
                }
            }
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    pub fn finish(mut self, model_name: &str) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }

        let mut stdout = io::stdout();
        execute!(
            stdout,
            MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            SetForegroundColor(Theme::SUCCESS_GREEN),
            SetAttribute(Attribute::Bold),
            Print("✓ "),
            ResetColor,
            SetForegroundColor(Theme::TEXT_PRIMARY),
            Print(model_name),
            ResetColor,
            Print("\n")
        )
        .ok();
    }

    pub fn finish_with_error(mut self, message: &str) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }

        let mut stdout = io::stdout();
        execute!(
            stdout,
            MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            SetForegroundColor(Theme::ERROR_RED),
            SetAttribute(Attribute::Bold),
            Print("✗ "),
            ResetColor,
            SetForegroundColor(Theme::TEXT_PRIMARY),
            Print(message),
            ResetColor,
            Print("\n")
        )
        .ok();
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ModelLoader {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

pub fn print_model_info(
    name: &str,
    size: &str,
    quant: &str,
    layers: usize,
    dim: usize,
    context: usize,
) {
    let mut stdout = io::stdout();

    execute!(
        stdout,
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  ├─ "),
        ResetColor,
        Print("Model:   "),
        SetForegroundColor(Theme::TEXT_PRIMARY),
        Print(name),
        ResetColor,
        Print("\n"),
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  ├─ "),
        ResetColor,
        Print("Size:    "),
        SetForegroundColor(Theme::TEXT_SECONDARY),
        Print(size),
        ResetColor,
        Print("\n"),
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  ├─ "),
        ResetColor,
        Print("Quant:   "),
        SetForegroundColor(Theme::FERRIS_ORANGE),
        Print(quant),
        ResetColor,
        Print("\n"),
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  ├─ "),
        ResetColor,
        Print("Context: "),
        SetForegroundColor(Theme::TEXT_SECONDARY),
        Print(format!("{} tokens", context)),
        ResetColor,
        Print("\n"),
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  └─ "),
        ResetColor,
        Print("Layers:  "),
        SetForegroundColor(Theme::TEXT_SECONDARY),
        Print(format!("{} (dim: {})", layers, dim)),
        ResetColor,
        Print("\n\n")
    )
    .ok();
}
