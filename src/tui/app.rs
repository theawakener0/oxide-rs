use std::path::PathBuf;
use std::sync::{mpsc, Mutex};
use std::thread;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::{backend::CrosstermBackend, Frame, Terminal};

use crate::tui::components::input::InputWidget;
use crate::tui::components::notification::Notification;
use crate::tui::components::sidebar::Sidebar;
use crate::tui::components::status_bar::StatusBar;
use crate::tui::screens::chat::ChatScreen;
use crate::tui::screens::models::ModelsScreen;
use crate::tui::screens::settings::SettingsScreen;
use crate::tui::state::{AppState, FocusArea, NotificationLevel, PendingAction, Screen};
use crate::{list_models, unregister_model, GenerateOptions, Model};

static APP_STATE: Mutex<Option<AppState>> = Mutex::new(None);

enum WorkerCommand {
    LoadModel {
        path: PathBuf,
        options: GenerateOptions,
    },
    UpdateOptions {
        options: GenerateOptions,
        reload_model: bool,
    },
    Generate {
        prompt: String,
    },
    Download {
        repo_id: String,
    },
    Shutdown,
}

enum WorkerEvent {
    ModelLoadStarted(String),
    ModelLoaded {
        path: PathBuf,
    },
    ContextUpdated {
        used: usize,
        limit: usize,
    },
    FirstToken,
    Token(String),
    GenerationStarted,
    GenerationFinished,
    DownloadStarted(String),
    DownloadProgress {
        filename: String,
        bytes_downloaded: u64,
        total_bytes: u64,
    },
    DownloadCompleted(String),
    DownloadError(String),
    Error(String),
}

const SETTINGS_FIELD_COUNT: usize = 8;

pub struct App {
    terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
    input: InputWidget,
    should_quit: bool,
    worker_tx: mpsc::Sender<WorkerCommand>,
    worker_rx: mpsc::Receiver<WorkerEvent>,
}

impl App {
    pub fn new(model_path: Option<PathBuf>, initial_screen: Option<Screen>) -> Self {
        let backend = CrosstermBackend::new(std::io::stdout());
        let mut terminal = Terminal::new(backend).expect("Failed to create terminal");
        terminal.clear().expect("Failed to clear terminal");

        let mut state = AppState::new();
        state.model_path = model_path.clone();

        if let Some(screen) = initial_screen {
            state.set_current_screen(screen);
        }

        if let Some(path) = &model_path {
            if let Ok(models) = crate::list_models() {
                if let Some(index) = models.iter().position(|model| &model.path == path) {
                    state.selected_model_index = index;
                }
            }
        }
        *APP_STATE.lock().unwrap() = Some(state);

        let (worker_tx, worker_cmd_rx) = mpsc::channel();
        let (worker_event_tx, worker_rx) = mpsc::channel();

        thread::spawn(move || Self::worker_loop(worker_cmd_rx, worker_event_tx));

        let mut app = Self {
            terminal,
            input: InputWidget::new(),
            should_quit: false,
            worker_tx,
            worker_rx,
        };

        if let Some(path) = model_path {
            app.load_model(path);
        }

        app
    }

    pub fn current_state() -> AppState {
        APP_STATE.lock().unwrap().clone().unwrap_or_default()
    }

    pub fn state_mut() -> std::sync::MutexGuard<'static, Option<AppState>> {
        APP_STATE.lock().unwrap()
    }

    pub fn run(&mut self) -> Result<()> {
        loop {
            self.process_worker_events();
            {
                let mut state_guard = Self::state_mut();
                if let Some(state) = state_guard.as_mut() {
                    state.clear_expired_notification();
                }
            }

            let state = Self::current_state();
            let input = self.input.clone();
            self.terminal.draw(|f| Self::render(f, &state, &input))?;

            if self.should_quit {
                let _ = self.worker_tx.send(WorkerCommand::Shutdown);
                break;
            }

            self.handle_input()?;
        }

        Ok(())
    }

    fn worker_loop(rx: mpsc::Receiver<WorkerCommand>, tx: mpsc::Sender<WorkerEvent>) {
        let mut model: Option<Model> = None;
        let mut current_path: Option<PathBuf> = None;

        while let Ok(command) = rx.recv() {
            match command {
                WorkerCommand::LoadModel { path, options } => {
                    let label = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model")
                        .to_string();
                    let _ = tx.send(WorkerEvent::ModelLoadStarted(label));

                    match Model::new(&path).map(|m| m.with_options(options)) {
                        Ok(mut loaded) => match loaded.load() {
                            Ok(_) => {
                                let used = loaded.context_used().unwrap_or(0);
                                let limit = loaded.context_limit().unwrap_or(0);
                                current_path = Some(path.clone());
                                model = Some(loaded);
                                let _ = tx.send(WorkerEvent::ModelLoaded { path });
                                let _ = tx.send(WorkerEvent::ContextUpdated { used, limit });
                            }
                            Err(err) => {
                                let _ = tx.send(WorkerEvent::Error(format!(
                                    "Failed to load model: {}",
                                    err
                                )));
                            }
                        },
                        Err(err) => {
                            let _ = tx.send(WorkerEvent::Error(format!(
                                "Failed to initialize model: {}",
                                err
                            )));
                        }
                    }
                }
                WorkerCommand::Generate { prompt } => {
                    let Some(active_model) = model.as_mut() else {
                        let _ = tx.send(WorkerEvent::Error("No model loaded".to_string()));
                        continue;
                    };

                    if current_path.is_some() {
                        let _ = tx.send(WorkerEvent::GenerationStarted);
                    }

                    let _ = tx.send(WorkerEvent::ContextUpdated {
                        used: active_model.context_used().unwrap_or(0),
                        limit: active_model.context_limit().unwrap_or(0),
                    });

                    let mut saw_first_token = false;
                    match active_model.generate_stream(&prompt, |token| {
                        if !saw_first_token {
                            saw_first_token = true;
                            let _ = tx.send(WorkerEvent::FirstToken);
                        }
                        let _ = tx.send(WorkerEvent::Token(token));
                    }) {
                        Ok(_) => {
                            let _ = tx.send(WorkerEvent::ContextUpdated {
                                used: active_model.context_used().unwrap_or(0),
                                limit: active_model.context_limit().unwrap_or(0),
                            });
                            let _ = tx.send(WorkerEvent::GenerationFinished);
                        }
                        Err(err) => {
                            let _ =
                                tx.send(WorkerEvent::Error(format!("Generation error: {}", err)));
                        }
                    }
                }
                WorkerCommand::UpdateOptions {
                    options,
                    reload_model,
                } => {
                    if reload_model {
                        let Some(path) = current_path.clone() else {
                            let _ = tx.send(WorkerEvent::Error("No model loaded".to_string()));
                            continue;
                        };

                        let label = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("model")
                            .to_string();
                        let _ = tx.send(WorkerEvent::ModelLoadStarted(label));

                        match Model::new(&path).map(|m| m.with_options(options)) {
                            Ok(mut loaded) => match loaded.load() {
                                Ok(_) => {
                                    let used = loaded.context_used().unwrap_or(0);
                                    let limit = loaded.context_limit().unwrap_or(0);
                                    model = Some(loaded);
                                    let _ = tx.send(WorkerEvent::ModelLoaded { path });
                                    let _ = tx.send(WorkerEvent::ContextUpdated { used, limit });
                                }
                                Err(err) => {
                                    let _ = tx.send(WorkerEvent::Error(format!(
                                        "Failed to reload model: {}",
                                        err
                                    )));
                                }
                            },
                            Err(err) => {
                                let _ = tx.send(WorkerEvent::Error(format!(
                                    "Failed to initialize model: {}",
                                    err
                                )));
                            }
                        }
                    } else if let Some(path) = current_path.clone() {
                        model = Model::new(&path).ok().map(|m| m.with_options(options));
                        if let Some(active_model) = model.as_mut() {
                            if active_model.load().is_ok() {
                                let _ = tx.send(WorkerEvent::ContextUpdated {
                                    used: active_model.context_used().unwrap_or(0),
                                    limit: active_model.context_limit().unwrap_or(0),
                                });
                            }
                        }
                    }
                }
                WorkerCommand::Download { repo_id } => {
                    let repo_id_clone = repo_id.clone();
                    let _ = tx.send(WorkerEvent::DownloadStarted(repo_id.clone()));

                    match crate::model::download_model(&repo_id, None, |progress| {
                        let _ = tx.send(WorkerEvent::DownloadProgress {
                            filename: progress.filename.clone(),
                            bytes_downloaded: progress.bytes_downloaded,
                            total_bytes: progress.total_bytes,
                        });
                    }) {
                        Ok((path, filename)) => {
                            if let Err(e) = crate::register_model(&repo_id, &filename, path, 0) {
                                let _ = tx.send(WorkerEvent::DownloadError(format!(
                                    "Failed to register model: {}",
                                    e
                                )));
                            } else {
                                let _ = tx.send(WorkerEvent::DownloadCompleted(repo_id_clone));
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(WorkerEvent::DownloadError(format!(
                                "Download failed: {}",
                                e
                            )));
                        }
                    }
                }
                WorkerCommand::Shutdown => break,
            }
        }
    }

    fn process_worker_events(&mut self) {
        while let Ok(event) = self.worker_rx.try_recv() {
            match event {
                WorkerEvent::ModelLoadStarted(name) => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.is_loading_model = true;
                        state.set_notification(
                            NotificationLevel::Info,
                            format!("Loading model: {}", name),
                        );
                    }
                }
                WorkerEvent::ModelLoaded { path } => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.model_path = Some(path.clone());
                        state.is_loading_model = false;
                        state.options = state.draft_options.clone();
                        state.settings_dirty = false;
                        state.set_notification(
                            NotificationLevel::Success,
                            format!(
                                "Model ready: {}",
                                path.file_name().and_then(|s| s.to_str()).unwrap_or("model")
                            ),
                        );
                    }
                }
                WorkerEvent::ContextUpdated { used, limit } => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.context_used = used;
                        state.context_limit = limit;
                    }
                }
                WorkerEvent::GenerationStarted => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.is_generating = true;
                        state.tokens_generated = 0;
                        state.tokens_per_second = 0.0;
                        state.clear_notification();
                    }
                }
                WorkerEvent::FirstToken => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.finish_thinking();
                    }
                }
                WorkerEvent::Token(token) => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.append_to_last_message(&token);
                        state.tokens_generated += 1;
                        let elapsed = state
                            .messages
                            .last()
                            .map(|msg| msg.timestamp.elapsed().as_secs_f64())
                            .unwrap_or(0.001);
                        state.tokens_per_second =
                            state.tokens_generated as f64 / elapsed.max(0.001);
                    }
                }
                WorkerEvent::GenerationFinished => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.finish_thinking();
                        state.is_generating = false;
                    }
                }
                WorkerEvent::DownloadStarted(repo_id) => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.download_state =
                            crate::tui::state::DownloadState::FetchingInfo(repo_id);
                        state.set_notification(NotificationLevel::Info, "Fetching model info...");
                    }
                }
                WorkerEvent::DownloadProgress {
                    filename,
                    bytes_downloaded,
                    total_bytes,
                } => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.download_state = crate::tui::state::DownloadState::Downloading {
                            repo_id: String::new(),
                            filename: filename.clone(),
                            bytes_downloaded,
                            total_bytes,
                        };
                        let percent = if total_bytes > 0 {
                            (bytes_downloaded as f64 / total_bytes as f64 * 100.0) as u32
                        } else {
                            0
                        };
                        let downloaded_mb = bytes_downloaded as f64 / 1_000_000.0;
                        let total_mb = total_bytes as f64 / 1_000_000.0;
                        state.set_notification(
                            NotificationLevel::Info,
                            format!(
                                "Downloading: {:.1}/{:.1} MB ({}%)",
                                downloaded_mb, total_mb, percent
                            ),
                        );
                    }
                }
                WorkerEvent::DownloadCompleted(repo_id) => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.download_state =
                            crate::tui::state::DownloadState::Completed(repo_id.clone());
                        state.set_notification(
                            NotificationLevel::Success,
                            format!("Downloaded: {}", repo_id),
                        );
                        state.clear_pending_action();
                    }
                }
                WorkerEvent::DownloadError(message) => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.download_state =
                            crate::tui::state::DownloadState::Error(message.clone());
                        state.set_notification(NotificationLevel::Error, message);
                        state.clear_pending_action();
                    }
                }
                WorkerEvent::Error(message) => {
                    let mut state_guard = Self::state_mut();
                    if let Some(state) = state_guard.as_mut() {
                        state.finish_thinking();
                        state.is_generating = false;
                        state.is_loading_model = false;
                        state.set_notification(NotificationLevel::Error, message);
                    }
                }
            }
        }
    }

    fn load_model(&mut self, path: PathBuf) {
        let options = Self::current_state().options;
        {
            let mut state_guard = Self::state_mut();
            if let Some(state) = state_guard.as_mut() {
                state.is_loading_model = true;
                state.model_path = Some(path.clone());
                state.clear_notification();
            }
        }
        let _ = self
            .worker_tx
            .send(WorkerCommand::LoadModel { path, options });
    }

    fn handle_input(&mut self) -> Result<()> {
        if !event::poll(std::time::Duration::from_millis(16))? {
            return Ok(());
        }

        let Event::Key(key) = event::read()? else {
            return Ok(());
        };

        if key.kind != KeyEventKind::Press {
            return Ok(());
        }

        if matches!(key.code, KeyCode::Char('?') | KeyCode::F(1)) {
            let mut state_guard = Self::state_mut();
            if let Some(state) = state_guard.as_mut() {
                state.show_help = !state.show_help;
            }
            return Ok(());
        }

        let state = Self::current_state();
        if state.show_help {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('?') | KeyCode::F(1)) {
                let mut state_guard = Self::state_mut();
                if let Some(state) = state_guard.as_mut() {
                    state.show_help = false;
                }
            }
            return Ok(());
        }

        if state.focus_area == crate::tui::state::FocusArea::Sidebar {
            return self.handle_sidebar_input(key.code);
        }

        match state.current_screen {
            Screen::Chat => self.handle_chat_input(key.code),
            Screen::Models => self.handle_models_input(key.code),
            Screen::Settings => self.handle_settings_input(key.code),
        }

        Ok(())
    }

    fn handle_sidebar_input(&mut self, code: KeyCode) -> Result<()> {
        let mut state_guard = Self::state_mut();
        let state = state_guard.as_mut().unwrap();

        match code {
            KeyCode::Tab => state.cycle_focus_forward(),
            KeyCode::BackTab => state.cycle_focus_backward(),
            KeyCode::Down | KeyCode::Char('j') => state.select_next_screen(),
            KeyCode::Up | KeyCode::Char('k') => state.select_prev_screen(),
            KeyCode::Enter => {
                let screen = state.sidebar_selected_screen;
                state.set_current_screen(screen);
            }
            KeyCode::Esc => {
                if state.notification.is_some() {
                    state.clear_notification();
                } else {
                    self.should_quit = true;
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_chat_input(&mut self, code: KeyCode) {
        let mut state_guard = Self::state_mut();
        let state = state_guard.as_mut().unwrap();

        if state.focus_area == FocusArea::Main {
            match code {
                KeyCode::Tab => state.cycle_focus_forward(),
                KeyCode::BackTab => state.cycle_focus_backward(),
                KeyCode::Down | KeyCode::Char('j') => {
                    state.chat_scroll = state.chat_scroll.saturating_add(1);
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    state.chat_scroll = state.chat_scroll.saturating_sub(1);
                }
                KeyCode::Esc => {
                    if state.notification.is_some() {
                        state.clear_notification();
                    } else {
                        self.should_quit = true;
                    }
                }
                _ => {}
            }
            return;
        }

        if state.is_loading_model || state.is_generating {
            if matches!(code, KeyCode::Esc) {
                if state.notification.is_some() {
                    state.clear_notification();
                } else {
                    self.should_quit = true;
                }
            }
            return;
        }

        match code {
            KeyCode::Tab => state.cycle_focus_forward(),
            KeyCode::BackTab => state.cycle_focus_backward(),
            KeyCode::Esc => {
                if state.notification.is_some() {
                    state.clear_notification();
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Left => self.input.move_cursor_left(),
            KeyCode::Right => self.input.move_cursor_right(),
            KeyCode::Home => self.input.move_cursor_to_start(),
            KeyCode::End => self.input.move_cursor_to_end(),
            KeyCode::Backspace => self.input.delete_char(),
            KeyCode::Enter => {
                if !self.input.is_empty() {
                    let prompt = self.input.value().to_string();
                    self.input.clear();
                    state.add_user_message(&prompt);
                    state.chat_scroll = 0;
                    state.tokens_generated = 0;
                    state.tokens_per_second = 0.0;
                    state.start_assistant_message();

                    let _ = self.worker_tx.send(WorkerCommand::Generate { prompt });
                }
            }
            KeyCode::Char(c) => self.input.insert_char(c),
            _ => {}
        }
    }

    fn handle_models_input(&mut self, code: KeyCode) {
        let mut state_guard = Self::state_mut();
        let state = state_guard.as_mut().unwrap();

        if let Some(pending_action) = &state.pending_action {
            match code {
                KeyCode::Char('y') | KeyCode::Enter => {
                    if let crate::tui::state::PendingAction::RemoveModel(ref model_id) =
                        *pending_action
                    {
                        let model_id = model_id.clone();
                        let active_model_path = state.model_path.clone();
                        drop(state_guard);

                        let was_active = if let Some(ref active_path) = active_model_path {
                            if let Ok(models) = list_models() {
                                models
                                    .iter()
                                    .any(|m| m.id == model_id && m.path == *active_path)
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        match unregister_model(&model_id) {
                            Ok(Some(_)) => {
                                let mut state_guard = Self::state_mut();
                                if was_active {
                                    state_guard.as_mut().unwrap().model_path = None;
                                    state_guard.as_mut().unwrap().context_used = 0;
                                    state_guard.as_mut().unwrap().context_limit = 0;
                                }
                                state_guard.as_mut().unwrap().set_notification(
                                    NotificationLevel::Success,
                                    format!("Removed model: {}", model_id),
                                );
                                state_guard.as_mut().unwrap().clear_pending_action();
                            }
                            Ok(None) => {
                                let mut state_guard = Self::state_mut();
                                state_guard.as_mut().unwrap().set_notification(
                                    NotificationLevel::Error,
                                    format!("Model not found: {}", model_id),
                                );
                                state_guard.as_mut().unwrap().clear_pending_action();
                            }
                            Err(err) => {
                                let mut state_guard = Self::state_mut();
                                state_guard.as_mut().unwrap().set_notification(
                                    NotificationLevel::Error,
                                    format!("Failed to remove model: {}", err),
                                );
                                state_guard.as_mut().unwrap().clear_pending_action();
                            }
                        }
                    }
                    return;
                }
                KeyCode::Char('n') | KeyCode::Esc => {
                    state.clear_pending_action();
                    state.set_notification(NotificationLevel::Info, "Remove cancelled");
                    return;
                }
                _ => return,
            }
        }

        if state.focus_area == FocusArea::DownloadInput {
            match code {
                KeyCode::Tab => {
                    state.cycle_focus_forward();
                    return;
                }
                KeyCode::BackTab => {
                    state.cycle_focus_backward();
                    return;
                }
                KeyCode::Esc => {
                    state.cancel_download();
                    state.focus_area = FocusArea::Main;
                    return;
                }
                KeyCode::Enter => {
                    let repo_id = state.download_input.trim().to_string();
                    if !repo_id.is_empty() {
                        let _ = self.worker_tx.send(WorkerCommand::Download {
                            repo_id: repo_id.clone(),
                        });
                        state.download_input.clear();
                        state.focus_area = FocusArea::Main;
                    }
                    return;
                }
                KeyCode::Backspace => {
                    state.download_input.pop();
                    return;
                }
                KeyCode::Char(c) => {
                    state.download_input.push(c);
                    return;
                }
                _ => return,
            }
        }

        match code {
            KeyCode::Tab => state.cycle_focus_forward(),
            KeyCode::BackTab => state.cycle_focus_backward(),
            KeyCode::Down | KeyCode::Char('j') => {
                if let Ok(models) = crate::list_models() {
                    state.select_next_model(models.len());
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if let Ok(models) = crate::list_models() {
                    state.select_prev_model(models.len());
                }
            }
            KeyCode::Enter => {
                if let Ok(models) = crate::list_models() {
                    if let Some(model) = models.get(state.selected_model_index) {
                        let path = model.path.clone();
                        drop(state_guard);
                        self.load_model(path);
                        return;
                    }
                }
            }
            KeyCode::Char('x') => {
                if let Ok(models) = list_models() {
                    if let Some(model) = models.get(state.selected_model_index) {
                        state.request_remove_model(model.id.clone());
                    }
                }
            }
            KeyCode::Char('d') => {
                state.start_download_input();
                state.focus_area = FocusArea::DownloadInput;
            }
            KeyCode::Esc => {
                if state.notification.is_some() {
                    state.clear_notification();
                } else {
                    self.should_quit = true;
                }
            }
            _ => {}
        }
    }

    fn handle_settings_input(&mut self, code: KeyCode) {
        let mut state_guard = Self::state_mut();
        let state = state_guard.as_mut().unwrap();

        let editing_system_prompt = state.settings_selected_field == 7;

        if editing_system_prompt {
            match code {
                KeyCode::Tab => {
                    state.cycle_focus_forward();
                    return;
                }
                KeyCode::BackTab => {
                    state.cycle_focus_backward();
                    return;
                }
                KeyCode::Up => {
                    state.settings_selected_field = state.settings_selected_field.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    state.settings_selected_field =
                        (state.settings_selected_field + 1).min(SETTINGS_FIELD_COUNT - 1);
                    return;
                }
                KeyCode::Backspace => {
                    if let Some(prompt) = state.draft_options.system_prompt.as_mut() {
                        prompt.pop();
                    }
                    state.mark_settings_dirty();
                    return;
                }
                KeyCode::Enter => {
                    if state.settings_dirty {
                        state.options = state.draft_options.clone();
                        state.settings_dirty = false;
                        let options = state.options.clone();
                        let reload_model = true;
                        let _ = self.worker_tx.send(WorkerCommand::UpdateOptions {
                            options,
                            reload_model,
                        });
                        state.set_notification(NotificationLevel::Info, "Applying settings...");
                    }
                    return;
                }
                KeyCode::Esc => {
                    if state.notification.is_some() {
                        state.clear_notification();
                    } else {
                        self.should_quit = true;
                    }
                    return;
                }
                KeyCode::Char(c) => {
                    let prompt = state
                        .draft_options
                        .system_prompt
                        .get_or_insert_with(String::new);
                    prompt.push(c);
                    state.mark_settings_dirty();
                    return;
                }
                _ => {}
            }
        }

        match code {
            KeyCode::Tab => state.cycle_focus_forward(),
            KeyCode::BackTab => state.cycle_focus_backward(),
            KeyCode::Down | KeyCode::Char('j') => {
                state.settings_selected_field =
                    (state.settings_selected_field + 1).min(SETTINGS_FIELD_COUNT - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                state.settings_selected_field = state.settings_selected_field.saturating_sub(1);
            }
            KeyCode::Left | KeyCode::Char('h') => {
                adjust_setting(state, -1);
            }
            KeyCode::Right | KeyCode::Char('l') => {
                adjust_setting(state, 1);
            }
            KeyCode::Char('r') => {
                state.sync_draft_options();
            }
            KeyCode::Enter => {
                if state.settings_dirty {
                    state.options = state.draft_options.clone();
                    state.settings_dirty = false;
                    let options = state.options.clone();
                    let reload_model = true;
                    let _ = self.worker_tx.send(WorkerCommand::UpdateOptions {
                        options,
                        reload_model,
                    });
                    state.set_notification(NotificationLevel::Info, "Applying settings...");
                }
            }
            KeyCode::Esc => {
                if state.notification.is_some() {
                    state.clear_notification();
                } else {
                    self.should_quit = true;
                }
            }
            _ => {}
        }
    }

    fn render(f: &mut Frame, state: &AppState, input: &InputWidget) {
        let area = f.area();
        let screens = vec![Screen::Chat, Screen::Models, Screen::Settings];

        let sidebar_width = screens
            .iter()
            .map(|screen| (screen.label().len() + 4) as u16)
            .max()
            .unwrap_or(12)
            .max(14);

        let vertical = Layout::vertical([Constraint::Min(0), Constraint::Length(3)]).split(area);
        let horizontal =
            Layout::horizontal([Constraint::Length(sidebar_width), Constraint::Min(0)])
                .split(vertical[0]);

        f.render_widget(
            Sidebar::new(
                screens,
                state.current_screen,
                state.sidebar_selected_screen,
                state.focus_area,
                sidebar_width,
            ),
            horizontal[0],
        );

        let main_area = horizontal[1];
        match state.current_screen {
            Screen::Chat => {
                let chat_sections =
                    Layout::vertical([Constraint::Min(0), Constraint::Length(3)]).split(main_area);
                f.render_widget(ChatScreen::new(), chat_sections[0]);

                let mut input = input.clone();
                input.set_focused(!state.is_generating && !state.is_loading_model);
                f.render_widget(input, chat_sections[1]);
            }
            Screen::Models => f.render_widget(ModelsScreen::new(), main_area),
            Screen::Settings => f.render_widget(SettingsScreen::new(), main_area),
        }

        f.render_widget(StatusBar::new(state), vertical[1]);

        if let Some(notification) = &state.notification {
            let notif_area = ratatui::layout::Rect::new(
                area.x + area.width / 4,
                area.y + area.height / 4,
                area.width / 2,
                5,
            );
            let notif = match notification.level {
                NotificationLevel::Error => Notification::error(notification.message.clone()),
                NotificationLevel::Info => Notification::info(notification.message.clone()),
                NotificationLevel::Success => Notification::success(notification.message.clone()),
            };
            f.render_widget(notif, notif_area);
        }

        if state.show_help {
            let help_area = ratatui::layout::Rect::new(
                area.x + area.width / 6,
                area.y + area.height / 6,
                area.width * 2 / 3,
                area.height * 2 / 3,
            );
            let help = ratatui::widgets::Paragraph::new(
                "Shortcuts\n\nGlobal\n  F1 or ?   Toggle help\n  Tab       Next focus\n  Shift+Tab Previous focus\n  Esc       Dismiss/quit\n\nSidebar\n  j/k       Move screens\n  Enter     Open screen\n\nChat\n  Enter     Send prompt\n  j/k       Scroll chat (main focus)\n\nModels\n  j/k       Move models\n  Enter     Load model\n  x         Remove highlighted model\n\nSettings\n  j/k       Move field\n  h/l       Adjust value\n  Enter     Apply settings\n  r         Reset draft\n  Type      Edit system prompt field",
            )
            .block(
                ratatui::widgets::Block::bordered()
                    .border_type(ratatui::widgets::BorderType::Double)
                    .title(" Help "),
            );
            f.render_widget(help, help_area);
        }

        if let Some(pending_action) = &state.pending_action {
            if let PendingAction::RemoveModel(model_id) = pending_action {
                let modal_width = 50u16;
                let modal_height = 7u16;
                let modal_area = ratatui::layout::Rect::new(
                    area.x + (area.width.saturating_sub(modal_width)) / 2,
                    area.y + (area.height.saturating_sub(modal_height)) / 2,
                    modal_width,
                    modal_height,
                );

                let confirm_text = format!("Remove model '{}'?", model_id);
                let modal = ratatui::widgets::Paragraph::new(format!(
                    "{}\n\n  [y] Yes  [n] No",
                    confirm_text
                ))
                .block(
                    ratatui::widgets::Block::bordered()
                        .border_type(ratatui::widgets::BorderType::Double)
                        .title(" Confirm "),
                )
                .style(ratatui::style::Style::new().fg(ratatui::style::Color::Yellow));

                f.render_widget(modal, modal_area);
            }
        }
    }
}

fn adjust_setting(state: &mut AppState, direction: i32) {
    match state.settings_selected_field {
        0 => {
            state.draft_options.temperature =
                (state.draft_options.temperature + (direction as f64 * 0.1)).clamp(0.0, 2.0);
        }
        1 => {
            let current = state.draft_options.top_p.unwrap_or(1.0);
            let next = (current + (direction as f64 * 0.05)).clamp(0.05, 1.0);
            state.draft_options.top_p = Some(next);
        }
        2 => {
            let current = state.draft_options.top_k.unwrap_or(40) as i32;
            let next = (current + direction * 5).clamp(1, 200) as usize;
            state.draft_options.top_k = Some(next);
        }
        3 => {
            let current = state.draft_options.max_tokens as i32;
            state.draft_options.max_tokens = (current + direction * 32).clamp(32, 4096) as usize;
        }
        4 => {
            state.draft_options.repeat_penalty =
                (state.draft_options.repeat_penalty + (direction as f32 * 0.05)).clamp(1.0, 2.0);
        }
        5 => {
            let current = state.draft_options.repeat_last_n as i32;
            state.draft_options.repeat_last_n = (current + direction * 8).clamp(0, 512) as usize;
        }
        6 => {
            let current = state.draft_options.batch_size as i32;
            state.draft_options.batch_size = (current + direction * 8).clamp(8, 512) as usize;
        }
        7 => {}
        _ => {}
    }

    state.mark_settings_dirty();
}
