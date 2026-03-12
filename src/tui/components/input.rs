use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    widgets::{Block, Widget},
};

use crate::tui::theme::{IRON_GRAY, RUST_ORANGE, TEXT_PRIMARY};

#[derive(Clone)]
pub struct InputWidget {
    value: String,
    placeholder: String,
    cursor_position: usize,
    focused: bool,
}

impl InputWidget {
    pub fn new() -> Self {
        Self {
            value: String::new(),
            placeholder: "Type a message...".to_string(),
            cursor_position: 0,
            focused: false,
        }
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn set_value(&mut self, value: String) {
        self.value = value;
        self.cursor_position = self.value.len().min(self.cursor_position);
    }

    pub fn insert_char(&mut self, c: char) {
        if self.cursor_position <= self.value.len() {
            self.value.insert(self.cursor_position, c);
            self.cursor_position += 1;
        }
    }

    pub fn delete_char(&mut self) {
        if self.cursor_position > 0 && !self.value.is_empty() {
            self.value.remove(self.cursor_position - 1);
            self.cursor_position -= 1;
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_position < self.value.len() {
            self.cursor_position += 1;
        }
    }

    pub fn move_cursor_to_start(&mut self) {
        self.cursor_position = 0;
    }

    pub fn move_cursor_to_end(&mut self) {
        self.cursor_position = self.value.len();
    }

    pub fn clear(&mut self) {
        self.value.clear();
        self.cursor_position = 0;
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    pub fn is_empty(&self) -> bool {
        self.value.trim().is_empty()
    }
}

impl Default for InputWidget {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for InputWidget {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let border_style: Style = if self.focused {
            RUST_ORANGE.into()
        } else {
            IRON_GRAY.into()
        };

        let block = Block::bordered()
            .border_type(ratatui::widgets::BorderType::Thick)
            .border_style(border_style)
            .title(if self.focused { " > " } else { "   " });

        let content_area = block.inner(area);
        block.render(area, buf);
        let y = content_area.y;
        let display_text = if self.value.is_empty() && !self.focused {
            self.placeholder.clone()
        } else {
            self.value.clone()
        };

        let cursor_index = if self.focused {
            Some(self.cursor_position)
        } else {
            None
        };

        let mut x = content_area.x;
        for (i, c) in display_text.chars().enumerate() {
            if x >= content_area.x + content_area.width {
                break;
            }

            let style: Style = if self.value.is_empty() && !self.focused {
                TEXT_PRIMARY.into()
            } else {
                TEXT_PRIMARY.into()
            };

            buf[(x, y)].set_char(c).set_style(style);
            x += 1;

            if cursor_index == Some(i) && self.focused {
                if x < content_area.x + content_area.width {
                    buf[(x, y)].set_char('█').set_style(RUST_ORANGE);
                }
            }
        }

        if cursor_index == Some(display_text.len()) && self.focused {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char('█').set_style(RUST_ORANGE);
            }
        }
    }
}
