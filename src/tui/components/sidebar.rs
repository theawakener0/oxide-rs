use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{Block, Widget},
};

use crate::tui::state::{FocusArea, Screen};
use crate::tui::theme::{title, ACCENT_CYAN, IRON_GRAY, RUST_ORANGE, TEXT_SECONDARY};

pub struct Sidebar {
    screens: Vec<Screen>,
    active_screen: Screen,
    selected_screen: Screen,
    focused: bool,
    width: u16,
}

impl Sidebar {
    pub fn new(
        screens: Vec<Screen>,
        active_screen: Screen,
        selected_screen: Screen,
        focus_area: FocusArea,
        width: u16,
    ) -> Self {
        Self {
            screens,
            active_screen,
            selected_screen,
            focused: focus_area == FocusArea::Sidebar,
            width,
        }
    }

    fn calculate_width(&self) -> u16 {
        let max_label_len = self
            .screens
            .iter()
            .map(|s| s.icon().len() + 1 + s.label().len())
            .max()
            .unwrap_or(8);
        self.width.max(max_label_len as u16).max(8)
    }
}

impl Widget for Sidebar {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let width = self.calculate_width().min(area.width.saturating_sub(1));
        let sidebar_area = Rect::new(area.x, area.y, width, area.height);

        let block = Block::bordered()
            .border_type(ratatui::widgets::BorderType::Thick)
            .border_style(IRON_GRAY)
            .title(title(" oxide-rs "));

        let content_area = block.inner(sidebar_area);
        block.render(sidebar_area, buf);

        for (i, screen) in self.screens.iter().enumerate() {
            let y = content_area.y + i as u16;
            if y >= content_area.y + content_area.height {
                break;
            }

            let is_active = *screen == self.active_screen;
            let is_selected = *screen == self.selected_screen;
            let label = format!("{} {}", screen.icon(), screen.label());

            let x = content_area.x + 1;
            let style = if is_active {
                RUST_ORANGE
            } else if is_selected && self.focused {
                ACCENT_CYAN
            } else {
                TEXT_SECONDARY
            };

            for (j, c) in label.chars().enumerate() {
                let j = j as u16;
                if x + j < content_area.x + content_area.width {
                    buf[(x + j, y)].set_char(c).set_style(style);
                }
            }
        }
    }
}
