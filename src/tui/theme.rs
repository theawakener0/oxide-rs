use ratatui::style::Stylize;

pub const RUST_ORANGE: ratatui::style::Color = ratatui::style::Color::Rgb(206, 66, 43);
pub const FERRIS_ORANGE: ratatui::style::Color = ratatui::style::Color::Rgb(222, 165, 132);
pub const IRON_GRAY: ratatui::style::Color = ratatui::style::Color::Rgb(68, 68, 68);
pub const TEXT_PRIMARY: ratatui::style::Color = ratatui::style::Color::Rgb(235, 235, 235);
pub const TEXT_SECONDARY: ratatui::style::Color = ratatui::style::Color::Rgb(180, 180, 180);
pub const SUCCESS_GREEN: ratatui::style::Color = ratatui::style::Color::Rgb(80, 250, 123);
pub const ERROR_RED: ratatui::style::Color = ratatui::style::Color::Rgb(255, 85, 85);
pub const ACCENT_CYAN: ratatui::style::Color = ratatui::style::Color::Rgb(139, 233, 253);

pub fn title(text: &str) -> ratatui::text::Line<'_> {
    ratatui::text::Line::from(vec![text.fg(ratatui::style::Color::White).bold()])
}

pub fn secondary(text: &str) -> ratatui::text::Line<'_> {
    ratatui::text::Line::from(vec![text.fg(TEXT_SECONDARY)])
}
