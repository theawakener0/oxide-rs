use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{Block, Paragraph, Widget},
};

use crate::tui::state::NotificationLevel;
use crate::tui::theme::{ACCENT_CYAN, ERROR_RED, SUCCESS_GREEN};

pub struct Notification {
    message: String,
    notification_type: NotificationLevel,
}

impl Notification {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            notification_type: NotificationLevel::Error,
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            notification_type: NotificationLevel::Info,
        }
    }

    pub fn success(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            notification_type: NotificationLevel::Success,
        }
    }
}

impl Widget for Notification {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 20 || area.height < 3 {
            return;
        }

        let title = match self.notification_type {
            NotificationLevel::Error => " Error ",
            NotificationLevel::Info => " Info ",
            NotificationLevel::Success => " Success ",
        };

        let border_color = match self.notification_type {
            NotificationLevel::Error => ERROR_RED,
            NotificationLevel::Info => ACCENT_CYAN,
            NotificationLevel::Success => SUCCESS_GREEN,
        };

        let block = Block::bordered()
            .border_type(ratatui::widgets::BorderType::Double)
            .border_style(border_color)
            .title(title);

        let content_area = block.inner(area);
        block.render(area, buf);
        Paragraph::new(self.message).render(content_area, buf);
    }
}
