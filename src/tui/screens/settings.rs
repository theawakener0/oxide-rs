use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    widgets::{Block, Paragraph, Widget},
};

use crate::tui::state::FocusArea;
use crate::tui::theme::{ACCENT_CYAN, IRON_GRAY, RUST_ORANGE, TEXT_PRIMARY, TEXT_SECONDARY};

const SETTINGS_FIELDS: [&str; 8] = [
    "Temperature",
    "Top-p",
    "Top-k",
    "Max Tokens",
    "Repeat Penalty",
    "Repeat Last N",
    "Batch Size",
    "System Prompt",
];

pub struct SettingsScreen;

impl SettingsScreen {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SettingsScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for SettingsScreen {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::bordered()
            .border_type(ratatui::widgets::BorderType::Thick)
            .border_style(IRON_GRAY)
            .title(" Settings ");

        let content_area = block.inner(area);
        block.render(area, buf);

        let state = crate::tui::app::App::current_state();
        let active_model = state
            .model_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("none selected");

        let sections = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(content_area);

        Paragraph::new(format!(
            "Active Model: {}\nUse Up/Down to choose, Left/Right to change, Enter to apply, r to reset.",
            active_model
        ))
        .style(TEXT_SECONDARY)
        .render(sections[0], buf);

        let mut lines = Vec::new();
        let values = [
            format!("{:.2}", state.draft_options.temperature),
            state
                .draft_options
                .top_p
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "None".to_string()),
            state
                .draft_options
                .top_k
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".to_string()),
            state.draft_options.max_tokens.to_string(),
            format!("{:.2}", state.draft_options.repeat_penalty),
            state.draft_options.repeat_last_n.to_string(),
            state.draft_options.batch_size.to_string(),
            state
                .draft_options
                .system_prompt
                .as_deref()
                .unwrap_or("")
                .to_string(),
        ];

        for (idx, label) in SETTINGS_FIELDS.iter().enumerate() {
            let marker = if idx == state.settings_selected_field {
                ">"
            } else {
                " "
            };
            let color =
                if idx == state.settings_selected_field && state.focus_area == FocusArea::Main {
                    ACCENT_CYAN
                } else if state.settings_dirty {
                    RUST_ORANGE
                } else {
                    TEXT_PRIMARY
                };
            lines.push((format!("{} {:<16} {}", marker, label, values[idx]), color));
        }

        let mut y = sections[1].y;
        for (line, color) in lines {
            if y >= sections[1].y + sections[1].height {
                break;
            }

            for (idx, ch) in line.chars().enumerate() {
                let x = sections[1].x + idx as u16;
                if x >= sections[1].x + sections[1].width {
                    break;
                }
                buf[(x, y)].set_char(ch).set_style(color);
            }

            y += 1;
        }

        Paragraph::new(if state.settings_dirty {
            "Draft differs from active settings. Press Enter to apply or r to reset."
        } else {
            "Settings are in sync with the active model."
        })
        .style(TEXT_SECONDARY)
        .render(sections[2], buf);
    }
}
