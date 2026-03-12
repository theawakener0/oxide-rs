use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{Block, Paragraph, Widget},
};

use crate::tui::theme::{IRON_GRAY, TEXT_PRIMARY};

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

        let settings_text = format!(
            "Active Model:    {}\nTemperature:     {:.2}\nTop-p:           {}\nTop-k:           {}\nMax Tokens:      {}\nRepeat Penalty:  {:.2}\nRepeat Last N:   {}\nBatch Size:      {}\nSeed:            {}\n\nSystem Prompt:\n{}",
            active_model,
            state.options.temperature,
            state.options.top_p.map(|v| v.to_string()).unwrap_or_else(|| "None".to_string()),
            state.options.top_k.map(|v| v.to_string()).unwrap_or_else(|| "None".to_string()),
            state.options.max_tokens,
            state.options.repeat_penalty,
            state.options.repeat_last_n,
            state.options.batch_size,
            state.options.seed,
            state.options.system_prompt.as_deref().unwrap_or("None")
        );

        Paragraph::new(settings_text)
            .style(TEXT_PRIMARY)
            .wrap(ratatui::widgets::Wrap { trim: true })
            .render(content_area, buf);
    }
}
