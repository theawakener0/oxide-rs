use crossterm::event::KeyEvent;

pub struct InputWidget {
    text: String,
    cursor_position: usize,
}

impl InputWidget {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor_position: 0,
        }
    }

    pub fn get_text(&self) -> String {
        self.text.clone()
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor_position = 0;
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        use crossterm::event::KeyCode;

        match key.code {
            KeyCode::Char(c) => {
                if self.cursor_position == self.text.len() {
                    self.text.push(c);
                } else {
                    self.text.insert(self.cursor_position, c);
                }
                self.cursor_position += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_position > 0 {
                    self.cursor_position -= 1;
                    self.text.remove(self.cursor_position);
                }
            }
            KeyCode::Delete => {
                if self.cursor_position < self.text.len() {
                    self.text.remove(self.cursor_position);
                }
            }
            KeyCode::Left => {
                if self.cursor_position > 0 {
                    self.cursor_position -= 1;
                }
            }
            KeyCode::Right => {
                if self.cursor_position < self.text.len() {
                    self.cursor_position += 1;
                }
            }
            KeyCode::Home => {
                self.cursor_position = 0;
            }
            KeyCode::End => {
                self.cursor_position = self.text.len();
            }
            _ => {}
        }
    }
}

impl Default for InputWidget {
    fn default() -> Self {
        Self::new()
    }
}
