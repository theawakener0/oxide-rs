use std::io;

use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
};

pub fn print_banner() {
    let banner = r#"
   ____  ____________________
  / __ \/ ____/ ____/ ____/ |
 / / / / / __/ / __/ / __/  /|
/ /_/ / /_/ / /_/ / /___ / /
\____/\____/\____/_____//_/
"#;

    let lines = banner.trim_end().lines().collect::<Vec<_>>();
    let colors = [
        Color::Magenta,
        Color::Rgb {
            r: 200,
            g: 100,
            b: 200,
        },
        Color::Rgb {
            r: 150,
            g: 100,
            b: 220,
        },
        Color::Rgb {
            r: 100,
            g: 150,
            b: 230,
        },
        Color::Cyan,
    ];

    let mut stdout = io::stdout();

    execute!(stdout, Print("\n")).ok();

    for (i, line) in lines.iter().enumerate() {
        let color = colors[i % colors.len()];
        execute!(
            stdout,
            SetForegroundColor(color),
            Print(format!("{}\n", line)),
            ResetColor
        )
        .ok();
    }

    execute!(
        stdout,
        SetForegroundColor(Color::DarkGrey),
        Print("  Fast AI Inference"),
        ResetColor,
        Print(" • "),
        SetForegroundColor(Color::DarkGrey),
        Print("Rust Edition v0.1.0"),
        ResetColor,
        Print("\n\n")
    )
    .ok();
}
