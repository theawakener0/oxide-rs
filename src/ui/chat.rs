use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::Block,
    widgets::Widget,
    Frame,
};

#[derive(Clone)]
pub struct ChatWidget {
    messages: Vec<Message>,
    model_name: String,
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
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .title(format!("Chat - {}", self.model_name));

        let inner_area = block.inner(area);
        block.render(area, f.buffer_mut());

        let mut y = inner_area.y + inner_area.height - 1;

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
            f.buffer_mut()
                .set_string(inner_area.x, y, &role_text, role_style);

            let content_lines = wrap_text(
                &msg.content,
                (inner_area.width as usize).saturating_sub(role_text.len()),
            );

            for line in content_lines.iter().rev() {
                if y < inner_area.y {
                    break;
                }
                f.buffer_mut().set_string(
                    inner_area.x + role_text.len() as u16,
                    y,
                    line,
                    Style::default(),
                );
                y = y.saturating_sub(1);
            }

            y = y.saturating_sub(1);
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            messages: self.messages.clone(),
            model_name: self.model_name.clone(),
        }
    }
}

fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 || max_width < 10 {
        return vec![text.chars().take(200).collect()];
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.len() + word.len() + 1 > max_width {
            if !current_line.is_empty() {
                lines.push(current_line.clone());
                current_line.clear();
            }
        }

        if !current_line.is_empty() {
            current_line.push(' ');
        }
        current_line.push_str(word);
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
}
