use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::Clear,
    terminal::ClearType,
};

pub struct Spinner {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    message: String,
}

impl Spinner {
    pub fn new(message: &str) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let msg = message.to_string();
        let spinner_msg = msg.clone();

        let handle = thread::spawn({
            let running = running.clone();
            move || {
                let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
                let mut i = 0;
                let mut stdout = io::stdout();

                while running.load(Ordering::Relaxed) {
                    let frame = frames[i % frames.len()];
                    execute!(
                        stdout,
                        Clear(ClearType::CurrentLine),
                        Print(format!("\r  {} {} ", frame, spinner_msg)),
                        ResetColor
                    )
                    .ok();
                    stdout.flush().ok();
                    thread::sleep(Duration::from_millis(80));
                    i += 1;
                }
            }
        });

        Self {
            running,
            handle: Some(handle),
            message: msg,
        }
    }

    pub fn finish(mut self, message: &str) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }

        let mut stdout = io::stdout();
        execute!(
            stdout,
            Clear(ClearType::CurrentLine),
            Print("\r  "),
            SetForegroundColor(Color::Green),
            Print("✓"),
            ResetColor,
            Print(format!(" {}\n", message))
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
            Clear(ClearType::CurrentLine),
            Print("\r  "),
            SetForegroundColor(Color::Red),
            Print("✗"),
            ResetColor,
            Print(format!(" {}\n", message))
        )
        .ok();
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
