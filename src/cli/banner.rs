use std::io::{self, Write};
use std::thread;
use std::time::Duration;

use crossterm::{
    cursor::MoveTo,
    execute,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{Clear, ClearType},
};

use super::theme::Theme;

const FERRIS_BANNER: &str = r#"
  /$$$$$$            /$$       /$$          
 /$$__  $$          |__/      | $$          
| $$  \ $$ /$$   /$$ /$$  /$$$$$$$  /$$$$$$ 
| $$  | $$|  $$ /$$/| $$ /$$__  $$ /$$__  $$
| $$  | $$ \  $$$$/ | $$| $$  | $$| $$$$$$$$
| $$  | $$  >$$  $$ | $$| $$  | $$| $$_____/
|  $$$$$$/ /$$/\  $$| $$|  $$$$$$$|  $$$$$$$
 \______/ |__/  \__/|__/ \_______/ \_______/


                 🦀 OXIDE
              Fast AI Inference
"#;

pub fn print_banner() {
    let mut stdout = io::stdout();

    execute!(stdout, Clear(ClearType::All), MoveTo(0, 0)).ok();

    let lines: Vec<&str> = FERRIS_BANNER.lines().collect();

    let orange_shades: [Color; 14] = [
        Color::Rgb {
            r: 255,
            g: 107,
            b: 53,
        },
        Color::Rgb {
            r: 255,
            g: 120,
            b: 60,
        },
        Color::Rgb {
            r: 255,
            g: 133,
            b: 70,
        },
        Color::Rgb {
            r: 255,
            g: 146,
            b: 80,
        },
        Color::Rgb {
            r: 255,
            g: 159,
            b: 90,
        },
        Color::Rgb {
            r: 222,
            g: 165,
            b: 132,
        },
        Color::Rgb {
            r: 206,
            g: 145,
            b: 120,
        },
        Color::Rgb {
            r: 206,
            g: 130,
            b: 100,
        },
        Color::Rgb {
            r: 206,
            g: 115,
            b: 80,
        },
        Color::Rgb {
            r: 206,
            g: 100,
            b: 60,
        },
        Color::Rgb {
            r: 206,
            g: 85,
            b: 50,
        },
        Color::Rgb {
            r: 206,
            g: 70,
            b: 43,
        },
        Color::Rgb {
            r: 180,
            g: 60,
            b: 38,
        },
        Color::Rgb {
            r: 150,
            g: 50,
            b: 32,
        },
    ];

    for (i, line) in lines.iter().enumerate() {
        let color = if i < orange_shades.len() {
            orange_shades[i]
        } else {
            Theme::RUST_ORANGE
        };

        execute!(
            stdout,
            SetForegroundColor(color),
            Print(format!("{}\n", line)),
            ResetColor
        )
        .ok();

        thread::sleep(Duration::from_millis(25));
        stdout.flush().ok();
    }

    thread::sleep(Duration::from_millis(50));

    execute!(
        stdout,
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  "),
        ResetColor,
        SetForegroundColor(Theme::RUST_ORANGE),
        SetAttribute(Attribute::Bold),
        Print("v0.1.0"),
        ResetColor,
        Print("  •  "),
        SetForegroundColor(Theme::IRON_GRAY),
        Print("Press Ctrl+C to exit"),
        ResetColor,
        Print("\n\n")
    )
    .ok();

    stdout.flush().ok();
}

pub fn print_divider() {
    let mut stdout = io::stdout();
    execute!(
        stdout,
        SetForegroundColor(Theme::IRON_GRAY),
        Print("  "),
        Print("━".repeat(56)),
        Print("\n"),
        ResetColor
    )
    .ok();
}
