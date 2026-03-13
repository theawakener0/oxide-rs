use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{Block, Widget},
};

use crate::tui::state::{AppState, FocusArea};
use crate::tui::theme::{
    ACCENT_CYAN, ERROR_RED, IRON_GRAY, RUST_ORANGE, SUCCESS_GREEN, TEXT_SECONDARY,
};

pub struct StatusBar<'a> {
    state: &'a AppState,
}

impl<'a> StatusBar<'a> {
    pub fn new(state: &'a AppState) -> Self {
        Self { state }
    }
}

impl Widget for StatusBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::bordered()
            .border_type(ratatui::widgets::BorderType::Thick)
            .border_style(IRON_GRAY);

        let content_area = block.inner(area);
        block.render(area, buf);
        let y = content_area.y;
        let mut x = content_area.x + 1;

        let screen_text = format!(" {} ", self.state.current_screen.label());
        for c in screen_text.chars() {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char(c).set_style(RUST_ORANGE);
                x += 1;
            }
        }

        x += 1;

        let help_text = " F1:Help ";
        for c in help_text.chars() {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char(c).set_style(ACCENT_CYAN);
                x += 1;
            }
        }

        x += 1;

        let focus_text = match self.state.focus_area {
            FocusArea::Sidebar => " Focus:Nav ",
            FocusArea::Main => " Focus:Main ",
            FocusArea::Input => " Focus:Input ",
        };
        for c in focus_text.chars() {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char(c).set_style(ACCENT_CYAN);
                x += 1;
            }
        }

        x += 1;

        let model_name = self
            .state
            .model_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("No model");

        let model_text = format!(" {} ", model_name);
        for c in model_text.chars() {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char(c).set_style(TEXT_SECONDARY);
                x += 1;
            }
        }

        x += 1;

        let context_text = format!(
            " {}:{}/{} ",
            "Ctx", self.state.context_used, self.state.context_limit
        );
        for c in context_text.chars() {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char(c).set_style(ACCENT_CYAN);
                x += 1;
            }
        }

        x += 1;

        let tok_s_text = format!(" {:.1} tok/s ", self.state.tokens_per_second);
        for c in tok_s_text.chars() {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char(c).set_style(SUCCESS_GREEN);
                x += 1;
            }
        }

        x += 1;

        let temp_text = format!(" T:{:.1} ", self.state.options.temperature);
        for c in temp_text.chars() {
            if x < content_area.x + content_area.width {
                buf[(x, y)].set_char(c).set_style(TEXT_SECONDARY);
                x += 1;
            }
        }

        if self.state.settings_dirty {
            x += 1;
            let pending_text = " [pending apply] ";
            for c in pending_text.chars() {
                if x < content_area.x + content_area.width {
                    buf[(x, y)].set_char(c).set_style(ACCENT_CYAN);
                    x += 1;
                }
            }
        }

        if self.state.is_generating {
            x += 1;
            let gen_text = " [generating] ";
            for c in gen_text.chars() {
                if x < content_area.x + content_area.width {
                    buf[(x, y)].set_char(c).set_style(ERROR_RED);
                    x += 1;
                }
            }
        }
    }
}
