use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::Line,
    widgets::{Block, List, ListItem, Paragraph, Widget},
};

use crate::tui::app::App;
use crate::tui::state::FocusArea;
use crate::tui::theme::{
    ACCENT_CYAN, ERROR_RED, IRON_GRAY, RUST_ORANGE, TEXT_PRIMARY, TEXT_SECONDARY,
};
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

        let state = App::current_state();

        if state.focus_area == FocusArea::DownloadInput {
            let input_area = Rect::new(
                content_area.x + 2,
                content_area.y + content_area.height.saturating_sub(3),
                content_area.width.saturating_sub(4),
                3,
            );

            let input_text = format!("Download: {}▌", state.download_input);
            let input_block = Block::bordered()
                .border_style(ACCENT_CYAN)
                .title(" Enter repo-id ");

            Paragraph::new(input_text)
                .style(Style::new().fg(ACCENT_CYAN))
                .block(input_block)
                .render(input_area, buf);

            let hint_area = Rect::new(
                content_area.x + 2,
                content_area.y + content_area.height.saturating_sub(5),
                content_area.width.saturating_sub(4),
                1,
            );
            Paragraph::new("Enter: download  Esc: cancel  Tab: switch focus")
                .style(TEXT_SECONDARY)
                .render(hint_area, buf);
            return;
        }

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
                "No models downloaded yet.\n\nPress 'd' to download a model from HuggingFace.",
            )
            .style(TEXT_SECONDARY)
            .render(content_area, buf);
            return;
        }

        let sections = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(content_area);

        let items: Vec<ListItem> = models
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                let is_active = state.model_path.as_ref() == Some(&m.path);
                let marker = if idx == state.selected_model_index {
                    ">"
                } else {
                    " "
                };
                let active = if is_active { "*" } else { " " };
                let line = format!(
                    "{}{} {} ({})",
                    marker,
                    active,
                    m.id,
                    m.quantization.as_deref().unwrap_or("Unknown")
                );
                ListItem::new(Line::from(line))
            })
            .collect();

        List::new(items)
            .style(TEXT_PRIMARY)
            .render(sections[0], buf);

        if let Some(model) = models.get(state.selected_model_index) {
            let active = if state.model_path.as_ref() == Some(&model.path) {
                "Yes"
            } else {
                "No"
            };
            let preview = format!(
                "Highlighted\n\nID:          {}\nRepo:        {}\nFile:        {}\nQuant:       {}\nSize:        {}\nDownloaded:  {}\nActive:      {}\n\nActions\n\nEnter: load model\nx: remove model\nd: download new\nj/k: move selection\nTab: switch focus",
                model.id,
                model.repo_id,
                model.filename,
                model.quantization.as_deref().unwrap_or("Unknown"),
                format_size(model.size_bytes),
                model.downloaded_at.format("%Y-%m-%d %H:%M"),
                active,
            );

            let color = if state.model_path.as_ref() == Some(&model.path) {
                RUST_ORANGE
            } else {
                ACCENT_CYAN
            };

            Paragraph::new(preview)
                .style(color)
                .render(sections[1], buf);
        }
    }
}
