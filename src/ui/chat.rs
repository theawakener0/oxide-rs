use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Widget},
    Frame,
};

#[derive(Clone)]
pub struct ChatWidget {
    messages: Vec<Message>,
    model_name: String,
    streaming_message: Option<String>,
}

#[derive(Clone)]
struct Message {
    role: String,
    content: String,
}

impl ChatWidget {
    pub fn new(model_name: &str) -> Self {
        Self {
            messages: vec![],
            model_name: model_name.to_string(),
            streaming_message: None,
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
        self.streaming_message = None;
    }

    pub fn start_streaming(&mut self, role: &str) {
        self.streaming_message = Some(String::new());
        self.messages.push(Message {
            role: role.to_string(),
            content: String::new(),
        });
    }

    pub fn append_token(&mut self, token: &str) {
        if let Some(ref mut streaming) = self.streaming_message {
            streaming.push_str(token);
            if let Some(last) = self.messages.last_mut() {
                last.content.push_str(token);
            }
        }
    }

    pub fn finish_streaming(&mut self) {
        self.streaming_message = None;
    }

    pub fn is_streaming(&self) -> bool {
        self.streaming_message.is_some()
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .title(format!("Chat - {}", self.model_name));

        let inner_area = block.inner(area);
        block.render(area, f.buffer_mut());

        let mut y = inner_area.y + inner_area.height.saturating_sub(1);

        for msg in self.messages.iter().rev() {
            if y < inner_area.y {
                break;
            }

            let role_style = if msg.role == "User" {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Magenta)
            };

            let role_text = format!("{}: ", msg.role);
            let role_width = role_text.len() as u16;
            let available_width = inner_area.width;

            if role_width < available_width {
                f.buffer_mut()
                    .set_string(inner_area.x, y, &role_text, role_style);
            }

            let content_width = (available_width as usize).saturating_sub(role_width as usize);
            let content_lines = wrap_text(&msg.content, content_width);

            for line in content_lines.iter().rev() {
                if y < inner_area.y {
                    break;
                }
                let display_line = if line.len() > available_width as usize {
                    &line[..available_width as usize]
                } else {
                    line.as_str()
                };
                f.buffer_mut().set_string(
                    inner_area.x + role_width,
                    y,
                    display_line,
                    Style::default(),
                );
                y = y.saturating_sub(1);
            }

            y = y.saturating_sub(1);
        }
    }
}

fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 || max_width < 10 {
        let truncated: String = text.chars().take(200).collect();
        return vec![truncated];
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();

    for c in text.chars() {
        if c == '\n' {
            if !current_line.is_empty() {
                lines.push(current_line.clone());
                current_line.clear();
            }
            lines.push(String::new());
        } else if current_line.len() >= max_width {
            lines.push(current_line.clone());
            current_line.clear();
            current_line.push(c);
        } else {
            current_line.push(c);
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}
