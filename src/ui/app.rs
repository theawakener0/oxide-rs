use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
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
use crate::model::ModelLoader;

pub struct App {
    model_loader: ModelLoader,
    chat: ChatWidget,
    input: InputWidget,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    is_generating: Arc<AtomicBool>,
}

impl App {
    pub fn new(
        model_loader: ModelLoader,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repeat_penalty: f32,
    ) -> Self {
        let model_info = model_loader.get_model_info();

        Self {
            model_loader,
            chat: ChatWidget::new(&model_info.name),
            input: InputWidget::new(),
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            is_generating: Arc::new(AtomicBool::new(false)),
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

            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('c')
                            if key
                                .modifiers
                                .contains(crossterm::event::KeyModifiers::CONTROL) =>
                        {
                            if self.is_generating.load(Ordering::Relaxed) {
                                self.is_generating.store(false, Ordering::Relaxed);
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
        Ok(())
    }

    fn handle_submit(&mut self, prompt: String) {
        self.chat.add_message("User", &prompt);
        self.input.clear();

        let model_loader = ModelLoader::new(
            self.model_loader.model_path.clone(),
            self.model_loader.tokenizer_path.clone(),
        )
        .ok();

        let max_tokens = self.max_tokens;
        let temperature = self.temperature;
        let top_p = self.top_p;
        let top_k = self.top_k;
        let repeat_penalty = self.repeat_penalty;
        let is_generating = self.is_generating.clone();
        let mut chat = self.chat.clone();

        is_generating.store(true, Ordering::Relaxed);

        std::thread::spawn(move || {
            if let Some(ml) = model_loader {
                let result = ml.generate_streaming(
                    &prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    |_token, _is_done| {},
                );

                let final_response = result.unwrap_or_else(|e| format!("Error: {}", e));

                if is_generating.load(Ordering::Relaxed) {
                    chat.add_message("Assistant", &final_response);
                }
            }
        });
    }

    fn render(&mut self, f: &mut Frame) {
        let area = f.area();

        let model_info = self.model_loader.get_model_info();

        let header = Paragraph::new(format!(
            "oxide - {} ({})",
            model_info.name, model_info.param_count
        ))
        .style(Style::default().fg(Color::Cyan))
        .block(Block::default().borders(Borders::ALL).title("Model"));

        let footer = Paragraph::new("Ctrl+C: Exit  |  Enter: Send")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL));

        let status = if self.is_generating.load(Ordering::Relaxed) {
            " [Generating...]"
        } else {
            ""
        };

        let input_text = self.input.get_text();
        let input_block = Block::default()
            .borders(Borders::ALL)
            .title(format!("Input{}", status));

        let input_paragraph = Paragraph::new(input_text).block(input_block);

        f.render_widget(header, Rect::new(area.x, area.y, area.width, 3));
        f.render_widget(
            input_paragraph,
            Rect::new(area.x, area.height - 3, area.width, 3),
        );

        if self.is_generating.load(Ordering::Relaxed) {
            let gen_indicator =
                Paragraph::new("Generating response...").style(Style::default().fg(Color::Yellow));
            f.render_widget(
                gen_indicator,
                Rect::new(area.x, area.height - 4, area.width, 1),
            );
        }

        let chat_area = Rect::new(area.x, area.y + 3, area.width, area.height - 5);

        self.chat.render(f, chat_area);
        f.render_widget(footer, Rect::new(area.x, area.height - 1, area.width, 1));
    }
}
