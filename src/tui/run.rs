use std::path::PathBuf;

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use crate::tui::state::Screen;

pub fn run(
    model_path: Option<PathBuf>,
    download: Option<String>,
    initial_screen: Option<Screen>,
) -> Result<()> {
    if let Some(repo_id) = download {
        println!("Downloading model: {}", repo_id);
        crate::model::download_model(&repo_id, None, |_| {})?;
    }

    enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen)?;

    let mut app = crate::tui::App::new(model_path, initial_screen);
    let result = app.run();

    disable_raw_mode()?;
    execute!(std::io::stdout(), LeaveAlternateScreen)?;

    result?;

    Ok(())
}
