use std::path::PathBuf;

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

pub fn run(model_path: Option<PathBuf>, download: Option<String>) -> Result<()> {
    if let Some(repo_id) = download {
        println!("Downloading model: {}", repo_id);
        crate::model::download_model(&repo_id, None, |_| {})?;
    }

    enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen)?;

    let mut app = crate::tui::App::new(model_path);
    let result = app.run();

    disable_raw_mode()?;
    execute!(std::io::stdout(), LeaveAlternateScreen)?;

    result?;

    Ok(())
}
