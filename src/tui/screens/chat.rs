use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{Block, Paragraph, Widget},
};
use std::time::Duration;

use crate::tui::state::MessageRole;
use crate::tui::theme::{ACCENT_CYAN, FERRIS_ORANGE, RUST_ORANGE, TEXT_PRIMARY, TEXT_SECONDARY};

pub struct ChatScreen;

impl ChatScreen {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ChatScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for ChatScreen {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::bordered()
            .border_type(ratatui::widgets::BorderType::Thick)
            .border_style(crate::tui::theme::IRON_GRAY)
            .title(" Chat ");

        let content_area = block.inner(area);
        block.render(area, buf);
        let state = crate::tui::app::App::current_state();

        if state.messages.is_empty() {
            Paragraph::new(
                "Welcome to oxide-rs TUI!\n\nType your message and press Enter to start chatting.\n\nUse Tab to switch screens, Esc to quit.",
            )
            .style(TEXT_SECONDARY)
            .render(content_area, buf);
            return;
        }

        let mut lines = Vec::new();
        let active_model = state
            .model_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("none selected");
        lines.push((format!("Model: {}", active_model), TEXT_SECONDARY));
        lines.push((String::new(), TEXT_SECONDARY));

        for msg in state.messages.iter().rev().take(8).rev() {
            let header = match msg.role {
                MessageRole::User => ("You", RUST_ORANGE),
                MessageRole::Assistant => ("Assistant", ACCENT_CYAN),
            };
            lines.push((header.0.to_string(), header.1));
            if msg.is_thinking {
                let spinner = spinner_frame(msg.timestamp.elapsed());
                lines.push((format!("{} Thinking...", spinner), FERRIS_ORANGE));
            } else {
                for line in wrap_text(&msg.content, content_area.width.saturating_sub(1) as usize) {
                    lines.push((line, TEXT_PRIMARY));
                }
            }
            lines.push((String::new(), TEXT_SECONDARY));
        }

        let mut y = content_area.y;
        for (line, color) in lines {
            if y >= content_area.y + content_area.height {
                break;
            }
            for (idx, ch) in line.chars().enumerate() {
                let x = content_area.x + idx as u16;
                if x >= content_area.x + content_area.width {
                    break;
                }
                buf[(x, y)].set_char(ch).set_style(color);
            }
            y += 1;
        }
    }
}

fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.len() + 1 + word.len() > max_width {
            lines.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn spinner_frame(elapsed: Duration) -> &'static str {
    const FRAMES: [&str; 8] = ["-", "\\", "|", "/", "-", "\\", "|", "/"];
    let frame = ((elapsed.as_millis() / 120) as usize) % FRAMES.len();
    FRAMES[frame]
}
