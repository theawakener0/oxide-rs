use crossterm::style::Color;

pub struct Theme;

impl Theme {
    pub const RUST_ORANGE: Color = Color::Rgb {
        r: 206,
        g: 66,
        b: 43,
    };
    pub const FERRIS_ORANGE: Color = Color::Rgb {
        r: 222,
        g: 165,
        b: 132,
    };
    pub const IRON_GRAY: Color = Color::Rgb {
        r: 68,
        g: 68,
        b: 68,
    };
    pub const TEXT_PRIMARY: Color = Color::Rgb {
        r: 235,
        g: 235,
        b: 235,
    };
    pub const TEXT_SECONDARY: Color = Color::Rgb {
        r: 180,
        g: 180,
        b: 180,
    };
    pub const SUCCESS_GREEN: Color = Color::Rgb {
        r: 80,
        g: 250,
        b: 123,
    };
    pub const ERROR_RED: Color = Color::Rgb {
        r: 255,
        g: 85,
        b: 85,
    };
    pub const ACCENT_CYAN: Color = Color::Rgb {
        r: 139,
        g: 233,
        b: 253,
    };
}
