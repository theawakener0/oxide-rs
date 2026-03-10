use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::{
    cursor::MoveToColumn,
    execute,
    style::{Attribute, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{Clear, ClearType},
};

use crate::cli::theme::Theme;
use crate::model::download::{format_size, DownloadProgress};

const DOWNLOAD_FRAMES: &[&str] = &[
    "⬇       ",
    "⬇  ⣀   ",
    "⬇ ⣀ ⣀  ",
    "⬇ ⣀ ⣀ ⣀",
    "⬇ ⣀ ⣀ ⣀",
    "⬇ ⣀ ⣀  ",
    "⬇  ⣀   ",
    "⬇       ",
];

const SPINNER_FRAMES: &[&str] = &["◐", "◑", "◒", "◓", "◑", "◐"];

pub struct DownloadProgressBar {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl DownloadProgressBar {
    pub fn new(filename: &str, _total_size: u64) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let filename_for_thread = filename.to_string();

        let handle = thread::spawn({
            let running = running.clone();
            move || {
                let mut stdout = io::stdout();
                let mut i = 0usize;

                loop {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }

                    let frame = DOWNLOAD_FRAMES[i % DOWNLOAD_FRAMES.len()];

                    execute!(
                        stdout,
                        MoveToColumn(0),
                        Clear(ClearType::CurrentLine),
                        SetForegroundColor(Theme::RUST_ORANGE),
                        Print(frame),
                        ResetColor,
                        Print(" "),
                        SetForegroundColor(Theme::TEXT_PRIMARY),
                        Print(&filename_for_thread),
                        ResetColor
                    )
                    .ok();

                    stdout.flush().ok();
                    thread::sleep(Duration::from_millis(80));
                    i = i.wrapping_add(1);
                }
            }
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    pub fn update(&mut self, progress: &DownloadProgress) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }

        let percentage = if progress.total_bytes > 0 {
            (progress.bytes_downloaded as f64 / progress.total_bytes as f64 * 100.0) as u32
        } else {
            0
        };

        let downloaded_str = format_size(progress.bytes_downloaded);
        let total_str = format_size(progress.total_bytes);

        let bar_width = 20;
        let filled = (bar_width as f64 * progress.bytes_downloaded as f64
            / (progress.total_bytes as f64).max(1.0)) as usize;
        let bar: String = format!(
            "{}{}",
            std::iter::repeat('█')
                .take(filled.min(bar_width))
                .collect::<String>(),
            std::iter::repeat('░')
                .take(bar_width - filled.min(bar_width))
                .collect::<String>()
        );

        let mut stdout = io::stdout();
        execute!(
            stdout,
            MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            SetForegroundColor(Theme::RUST_ORANGE),
            Print("⬇ "),
            ResetColor,
            SetForegroundColor(Theme::TEXT_PRIMARY),
            Print(&progress.filename),
            Print(" "),
            SetForegroundColor(Theme::ACCENT_CYAN),
            Print("│"),
            ResetColor,
            Print(" "),
            SetForegroundColor(Theme::SUCCESS_GREEN),
            Print(format!("[{}]", bar)),
            ResetColor,
            Print(" "),
            SetForegroundColor(Theme::TEXT_SECONDARY),
            Print(format!("{}%", percentage)),
            ResetColor,
            Print(" "),
            Print(format!("{}/{}", downloaded_str, total_str)),
        )
        .ok();
        stdout.flush().ok();

        self.running = Arc::new(AtomicBool::new(true));
        let filename = progress.filename.clone();
        let running = self.running.clone();

        self.handle = Some(thread::spawn({
            move || {
                let mut stdout = io::stdout();
                let mut i = 0usize;

                loop {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }

                    let frame = DOWNLOAD_FRAMES[i % DOWNLOAD_FRAMES.len()];

                    execute!(
                        stdout,
                        MoveToColumn(0),
                        Clear(ClearType::CurrentLine),
                        SetForegroundColor(Theme::RUST_ORANGE),
                        Print(frame),
                        ResetColor,
                        Print(" "),
                        SetForegroundColor(Theme::TEXT_PRIMARY),
                        Print(&filename),
                        ResetColor
                    )
                    .ok();

                    stdout.flush().ok();
                    thread::sleep(Duration::from_millis(80));
                    i = i.wrapping_add(1);
                }
            }
        }));
    }

    pub fn finish(mut self, path: &str) {
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
            Print("Download complete!"),
            ResetColor,
            Print("\n  "),
            SetForegroundColor(Theme::IRON_GRAY),
            Print("Saved to: "),
            SetForegroundColor(Theme::ACCENT_CYAN),
            Print(path),
            ResetColor,
            Print("\n\n")
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
            Print("\n\n")
        )
        .ok();
    }
}

pub struct Spinner {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    pub fn new(message: &str) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let message = message.to_string();

        let handle = thread::spawn({
            let running = running.clone();
            move || {
                let mut stdout = io::stdout();
                let mut i = 0usize;

                while running.load(Ordering::Relaxed) {
                    let frame = SPINNER_FRAMES[i % SPINNER_FRAMES.len()];

                    execute!(
                        stdout,
                        MoveToColumn(0),
                        Clear(ClearType::CurrentLine),
                        SetForegroundColor(Theme::RUST_ORANGE),
                        Print(frame),
                        ResetColor,
                        Print(" "),
                        SetForegroundColor(Theme::TEXT_PRIMARY),
                        Print(&message),
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

    pub fn finish(mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
    }

    pub fn finish_with_message(mut self, message: &str) {
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
            Print("✓ "),
            ResetColor,
            SetForegroundColor(Theme::TEXT_PRIMARY),
            Print(message),
            ResetColor,
            Print("\n")
        )
        .ok();
    }
}

impl Drop for DownloadProgressBar {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
