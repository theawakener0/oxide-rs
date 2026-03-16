use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::GenerateOptions;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Chat,
    Models,
    Settings,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FocusArea {
    Sidebar,
    Main,
    Input,
    DownloadInput,
}

impl Screen {
    pub fn next(self) -> Self {
        match self {
            Screen::Chat => Screen::Models,
            Screen::Models => Screen::Settings,
            Screen::Settings => Screen::Chat,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Screen::Chat => Screen::Settings,
            Screen::Models => Screen::Chat,
            Screen::Settings => Screen::Models,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Screen::Chat => "Chat",
            Screen::Models => "Models",
            Screen::Settings => "Settings",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    Error,
    Info,
    Success,
}

#[derive(Clone)]
pub struct NotificationState {
    pub level: NotificationLevel,
    pub message: String,
    pub shown_at: Instant,
    pub ttl: Option<Duration>,
}

#[derive(Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: Instant,
    pub is_thinking: bool,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            timestamp: Instant::now(),
            is_thinking: false,
        }
    }

    pub fn assistant() -> Self {
        Self {
            role: MessageRole::Assistant,
            content: String::new(),
            timestamp: Instant::now(),
            is_thinking: true,
        }
    }

    pub fn finish_thinking(&mut self) {
        self.is_thinking = false;
    }
}

#[derive(Clone)]
pub enum PendingAction {
    RemoveModel(String),
    #[allow(dead_code)]
    RemoveModelConfirmed(String),
}

#[derive(Clone, PartialEq)]
pub enum DownloadState {
    Idle,
    FetchingInfo(String),
    Downloading {
        repo_id: String,
        filename: String,
        bytes_downloaded: u64,
        total_bytes: u64,
    },
    Completed(String),
    Error(String),
}

#[derive(Clone)]
pub struct AppState {
    pub model_path: Option<PathBuf>,
    pub options: GenerateOptions,
    pub draft_options: GenerateOptions,
    pub messages: Vec<ChatMessage>,
    pub current_screen: Screen,
    pub sidebar_selected_screen: Screen,
    pub focus_area: FocusArea,
    pub selected_model_index: usize,
    pub settings_selected_field: usize,
    pub chat_scroll: usize,
    pub sidebar_width: u16,
    pub is_loading_model: bool,
    pub is_generating: bool,
    pub notification: Option<NotificationState>,
    pub show_help: bool,
    pub pending_action: Option<PendingAction>,
    pub settings_dirty: bool,
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub tokens_per_second: f64,
    pub context_used: usize,
    pub context_limit: usize,
    pub download_state: DownloadState,
    pub download_input: String,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model_path: None,
            options: GenerateOptions::default(),
            draft_options: GenerateOptions::default(),
            messages: Vec::new(),
            current_screen: Screen::Chat,
            sidebar_selected_screen: Screen::Chat,
            focus_area: FocusArea::Input,
            selected_model_index: 0,
            settings_selected_field: 0,
            chat_scroll: 0,
            sidebar_width: 0,
            is_loading_model: false,
            is_generating: false,
            notification: None,
            show_help: false,
            pending_action: None,
            settings_dirty: false,
            tokens_generated: 0,
            prompt_tokens: 0,
            tokens_per_second: 0.0,
            context_used: 0,
            context_limit: 4096,
            download_state: DownloadState::Idle,
            download_input: String::new(),
        }
    }

    pub fn context_percentage(&self) -> f32 {
        if self.context_limit == 0 {
            return 0.0;
        }
        (self.context_used as f32 / self.context_limit as f32) * 100.0
    }

    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::user(content));
    }

    pub fn start_assistant_message(&mut self) {
        self.messages.push(ChatMessage::assistant());
    }

    pub fn append_to_last_message(&mut self, token: &str) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == MessageRole::Assistant {
                msg.content.push_str(token);
            }
        }
    }

    pub fn finish_thinking(&mut self) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == MessageRole::Assistant {
                msg.finish_thinking();
            }
        }
    }

    pub fn set_notification(&mut self, level: NotificationLevel, message: impl Into<String>) {
        let ttl = match level {
            NotificationLevel::Error => None,
            NotificationLevel::Info => Some(Duration::from_secs(2)),
            NotificationLevel::Success => Some(Duration::from_secs(2)),
        };

        self.notification = Some(NotificationState {
            level,
            message: message.into(),
            shown_at: Instant::now(),
            ttl,
        });
    }

    pub fn clear_notification(&mut self) {
        self.notification = None;
    }

    pub fn request_remove_model(&mut self, model_id: String) {
        self.pending_action = Some(PendingAction::RemoveModel(model_id));
    }

    pub fn clear_pending_action(&mut self) {
        self.pending_action = None;
    }

    pub fn start_download_input(&mut self) {
        self.download_input.clear();
        self.download_state = DownloadState::Idle;
    }

    pub fn cancel_download(&mut self) {
        self.download_input.clear();
        self.download_state = DownloadState::Idle;
    }

    pub fn clear_expired_notification(&mut self) {
        let should_clear = self
            .notification
            .as_ref()
            .and_then(|notification| {
                notification
                    .ttl
                    .map(|ttl| notification.shown_at.elapsed() >= ttl)
            })
            .unwrap_or(false);

        if should_clear {
            self.notification = None;
        }
    }

    pub fn set_current_screen(&mut self, screen: Screen) {
        self.current_screen = screen;
        self.sidebar_selected_screen = screen;
        self.focus_area = match screen {
            Screen::Chat => FocusArea::Input,
            Screen::Models | Screen::Settings => FocusArea::Main,
        };
    }

    pub fn cycle_focus_forward(&mut self) {
        self.focus_area = match self.current_screen {
            Screen::Chat => match self.focus_area {
                FocusArea::Sidebar => FocusArea::Main,
                FocusArea::Main => FocusArea::Input,
                FocusArea::Input => FocusArea::Sidebar,
                FocusArea::DownloadInput => FocusArea::Sidebar,
            },
            Screen::Models | Screen::Settings => match self.focus_area {
                FocusArea::Sidebar => FocusArea::Main,
                FocusArea::Main => FocusArea::DownloadInput,
                FocusArea::DownloadInput | FocusArea::Input => FocusArea::Sidebar,
            },
        };
    }

    pub fn cycle_focus_backward(&mut self) {
        self.focus_area = match self.current_screen {
            Screen::Chat => match self.focus_area {
                FocusArea::Sidebar => FocusArea::Input,
                FocusArea::Input => FocusArea::Main,
                FocusArea::Main => FocusArea::Sidebar,
                FocusArea::DownloadInput => FocusArea::Main,
            },
            Screen::Models | Screen::Settings => match self.focus_area {
                FocusArea::Sidebar => FocusArea::DownloadInput,
                FocusArea::DownloadInput => FocusArea::Main,
                FocusArea::Main => FocusArea::Sidebar,
                FocusArea::Input => FocusArea::Sidebar,
            },
        };
    }

    pub fn select_next_screen(&mut self) {
        self.sidebar_selected_screen = self.sidebar_selected_screen.next();
    }

    pub fn select_prev_screen(&mut self) {
        self.sidebar_selected_screen = self.sidebar_selected_screen.prev();
    }

    pub fn mark_settings_dirty(&mut self) {
        self.settings_dirty = true;
    }

    pub fn sync_draft_options(&mut self) {
        self.draft_options = self.options.clone();
        self.settings_dirty = false;
    }

    pub fn select_next_model(&mut self, total: usize) {
        if total == 0 {
            self.selected_model_index = 0;
        } else {
            self.selected_model_index = (self.selected_model_index + 1).min(total - 1);
        }
    }

    pub fn select_prev_model(&mut self, total: usize) {
        if total == 0 {
            self.selected_model_index = 0;
        } else {
            self.selected_model_index = self.selected_model_index.saturating_sub(1);
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
