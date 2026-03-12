use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::Line,
    widgets::{Block, List, ListItem, Paragraph, Widget},
};

use crate::tui::app::App;
use crate::tui::theme::{ERROR_RED, IRON_GRAY, TEXT_PRIMARY, TEXT_SECONDARY};
use crate::{format_size, list_models};

pub struct ModelsScreen;

impl ModelsScreen {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ModelsScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for ModelsScreen {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::bordered()
            .border_type(ratatui::widgets::BorderType::Thick)
            .border_style(IRON_GRAY)
            .title(" Models ");

        let content_area = block.inner(area);
        block.render(area, buf);

        let models = match list_models() {
            Ok(models) => models,
            Err(_) => {
                Paragraph::new("Failed to load models")
                    .style(ERROR_RED)
                    .render(content_area, buf);
                return;
            }
        };

        if models.is_empty() {
            Paragraph::new(
                "No models downloaded yet.\n\nUse `oxide-rs --download <repo-id>` to download one.",
            )
            .style(TEXT_SECONDARY)
            .render(content_area, buf);
            return;
        }

        let state = App::current_state();

        let items: Vec<ListItem> = models
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                let quant = m.quantization.as_deref().unwrap_or("Unknown");
                let marker = if idx == state.selected_model_index {
                    ">"
                } else {
                    " "
                };
                let line = format!(
                    "{} {} ({}) - {}",
                    marker,
                    m.id,
                    format_size(m.size_bytes),
                    quant
                );
                ListItem::new(Line::from(line))
            })
            .collect();

        List::new(items)
            .style(TEXT_PRIMARY)
            .render(content_area, buf);
    }
}
