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
use crate::tui::state::{AppState, NotificationLevel, Screen};
use crate::{GenerateOptions, Model};

static APP_STATE: Mutex<Option<AppState>> = Mutex::new(None);

enum WorkerCommand {
    LoadModel {
        path: PathBuf,
        options: GenerateOptions,
    },
    Generate {
        prompt: String,
    },
    Shutdown,
}

enum WorkerEvent {
    ModelLoadStarted(String),
    ModelLoaded { path: PathBuf },
    FirstToken,
    Token(String),
    GenerationStarted,
    GenerationFinished,
    Error(String),
}

pub struct App {
    terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
    input: InputWidget,
    should_quit: bool,
    worker_tx: mpsc::Sender<WorkerCommand>,
    worker_rx: mpsc::Receiver<WorkerEvent>,
}

impl App {
    pub fn new(model_path: Option<PathBuf>) -> Self {
        let backend = CrosstermBackend::new(std::io::stdout());
        let mut terminal = Terminal::new(backend).expect("Failed to create terminal");
        terminal.clear().expect("Failed to clear terminal");

        let mut state = AppState::new();
        state.model_path = model_path.clone();
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
                                current_path = Some(path.clone());
                                model = Some(loaded);
                                let _ = tx.send(WorkerEvent::ModelLoaded { path });
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

                    let mut saw_first_token = false;
                    match active_model.generate_stream(&prompt, |token| {
                        if !saw_first_token {
                            saw_first_token = true;
                            let _ = tx.send(WorkerEvent::FirstToken);
                        }
                        let _ = tx.send(WorkerEvent::Token(token));
                    }) {
                        Ok(_) => {
                            let _ = tx.send(WorkerEvent::GenerationFinished);
                        }
                        Err(err) => {
                            let _ =
                                tx.send(WorkerEvent::Error(format!("Generation error: {}", err)));
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
                        state.set_notification(
                            NotificationLevel::Success,
                            format!(
                                "Model ready: {}",
                                path.file_name().and_then(|s| s.to_str()).unwrap_or("model")
                            ),
                        );
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

        let state = Self::current_state();
        match state.current_screen {
            Screen::Chat => self.handle_chat_input(key.code),
            Screen::Models => self.handle_models_input(key.code),
            Screen::Settings => self.handle_settings_input(key.code),
        }

        Ok(())
    }

    fn handle_chat_input(&mut self, code: KeyCode) {
        let mut state_guard = Self::state_mut();
        let state = state_guard.as_mut().unwrap();

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
            KeyCode::Tab => state.current_screen = state.current_screen.next(),
            KeyCode::BackTab => state.current_screen = state.current_screen.prev(),
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

        match code {
            KeyCode::Tab => state.current_screen = state.current_screen.next(),
            KeyCode::BackTab => state.current_screen = state.current_screen.prev(),
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

        match code {
            KeyCode::Tab => state.current_screen = state.current_screen.next(),
            KeyCode::BackTab => state.current_screen = state.current_screen.prev(),
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
            Sidebar::new(screens, state.current_screen, sidebar_width),
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
    }
}
