use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
    DefaultTerminal, Frame,
};

use super::chat::ChatWidget;
use super::input::InputWidget;
use crate::inference::{Generator, StreamEvent};
use crate::model::GgufMetadata;

pub struct App {
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    chat: ChatWidget,
    input: InputWidget,
    max_tokens: usize,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<usize>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
    max_history: usize,
    is_generating: Arc<AtomicBool>,
    cancel_flag: Arc<AtomicBool>,
    metadata: GgufMetadata,
    event_rx: Option<Receiver<StreamEvent>>,
    tokens_per_sec: Option<f64>,
}

impl App {
    pub fn new(
        generator: Generator,
        model_path: PathBuf,
        tokenizer_path: Option<PathBuf>,
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        max_history: usize,
    ) -> Self {
        let metadata = generator.metadata().clone();
        let model_name = metadata.name.clone();

        Self {
            model_path,
            tokenizer_path,
            chat: ChatWidget::new(&model_name),
            input: InputWidget::new(),
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            repeat_last_n,
            seed,
            max_history,
            is_generating: Arc::new(AtomicBool::new(false)),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            metadata,
            event_rx: None,
            tokens_per_sec: None,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = ratatui::Terminal::new(backend)?;

        let result = self.run_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(std::io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
        terminal.show_cursor()?;

        result
    }

    fn run_loop(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        loop {
            terminal.draw(|f| self.render(f))?;

            if let Ok(event) = event::poll(Duration::from_millis(16)) {
                if event {
                    if let Event::Key(key) = event::read()? {
                        if key.kind == KeyEventKind::Press {
                            match key.code {
                                KeyCode::Char('c')
                                    if key
                                        .modifiers
                                        .contains(crossterm::event::KeyModifiers::CONTROL) =>
                                {
                                    if self.is_generating.load(Ordering::Relaxed) {
                                        self.cancel_flag.store(true, Ordering::Relaxed);
                                        self.is_generating.store(false, Ordering::Relaxed);
                                        self.chat.finish_streaming();
                                        self.chat.add_message("System", "Generation cancelled");
                                    } else {
                                        break;
                                    }
                                }
                                KeyCode::Enter if !self.is_generating.load(Ordering::Relaxed) => {
                                    let prompt = self.input.get_text();
                                    if !prompt.is_empty() {
                                        self.handle_submit(prompt);
                                    }
                                }
                                _ => {
                                    self.input.handle_key(key);
                                }
                            }
                        }
                    }
                }
            }

            if let Some(ref rx) = self.event_rx {
                while let Ok(event) = rx.try_recv() {
                    match event {
                        StreamEvent::Token(t) => {
                            self.chat.append_token(&t);
                        }
                        StreamEvent::Done { tokens_generated } => {
                            self.chat.finish_streaming();
                            self.is_generating.store(false, Ordering::Relaxed);
                            self.tokens_per_sec = if tokens_generated > 0 {
                                Some(tokens_generated as f64)
                            } else {
                                None
                            };
                        }
                        StreamEvent::Error(e) => {
                            self.chat.finish_streaming();
                            self.is_generating.store(false, Ordering::Relaxed);
                            self.chat.add_message("Error", &e);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_submit(&mut self, prompt: String) {
        self.chat.add_message("User", &prompt);
        self.input.clear();

        let (tx, rx): (Sender<StreamEvent>, Receiver<StreamEvent>) = unbounded();
        self.event_rx = Some(rx);

        self.chat.start_streaming("Assistant");
        self.is_generating.store(true, Ordering::Relaxed);
        self.cancel_flag.store(false, Ordering::Relaxed);

        let model_path = self.model_path.clone();
        let tokenizer_path = self.tokenizer_path.clone();
        let max_tokens = self.max_tokens;
        let temperature = self.temperature;
        let top_p = self.top_p;
        let top_k = self.top_k;
        let repeat_penalty = self.repeat_penalty;
        let repeat_last_n = self.repeat_last_n;
        let seed = self.seed;
        let max_history = self.max_history;
        let cancel_flag = self.cancel_flag.clone();

        std::thread::spawn(move || {
            match Generator::new(
                &model_path,
                tokenizer_path.as_ref(),
                temperature,
                top_p,
                top_k,
                seed,
                max_history,
            ) {
                Ok(mut generator) => {
                    let result = generator.generate(
                        &prompt,
                        max_tokens,
                        repeat_penalty,
                        repeat_last_n,
                        |event| {
                            if cancel_flag.load(Ordering::Relaxed) {
                                return;
                            }
                            let _ = tx.send(event);
                        },
                    );

                    if let Err(e) = result {
                        let _ = tx.send(StreamEvent::Error(e.to_string()));
                    }
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!("Failed to load model: {}", e)));
                }
            }
        });
    }

    fn render(&mut self, f: &mut Frame) {
        let area = f.area();

        let header = Paragraph::new(format!(
            "oxide - {} ({})",
            self.metadata.name,
            format_size(self.metadata.file_size)
        ))
        .style(Style::default().fg(Color::Cyan))
        .block(Block::default().borders(Borders::ALL).title("Model"));

        let status_text = if self.is_generating.load(Ordering::Relaxed) {
            "Generating...".to_string()
        } else if let Some(tps) = self.tokens_per_sec {
            format!("Ready ({:.1} tok/s last)", tps)
        } else {
            "Ready".to_string()
        };

        let footer = Paragraph::new(format!(
            "Ctrl+C: {}  |  Enter: Send  |  {}",
            if self.is_generating.load(Ordering::Relaxed) {
                "Cancel"
            } else {
                "Exit"
            },
            status_text
        ))
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));

        let input_text = self.input.get_text();
        let input_block = Block::default().borders(Borders::ALL).title(
            if self.is_generating.load(Ordering::Relaxed) {
                "Input (generating...)"
            } else {
                "Input"
            },
        );

        let input_paragraph = Paragraph::new(input_text).block(input_block);

        f.render_widget(header, Rect::new(area.x, area.y, area.width, 3));
        f.render_widget(
            input_paragraph,
            Rect::new(area.x, area.height - 3, area.width, 3),
        );

        let chat_area = Rect::new(area.x, area.y + 3, area.width, area.height - 6);

        self.chat.render(f, chat_area);
        f.render_widget(footer, Rect::new(area.x, area.height - 1, area.width, 1));
    }
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
