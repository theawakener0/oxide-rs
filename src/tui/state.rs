use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::GenerateOptions;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Chat,
    Models,
    Settings,
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

    pub fn icon(&self) -> &'static str {
        match self {
            Screen::Chat => "💬",
            Screen::Models => "📦",
            Screen::Settings => "⚙",
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
pub struct AppState {
    pub model_path: Option<PathBuf>,
    pub options: GenerateOptions,
    pub messages: Vec<ChatMessage>,
    pub current_screen: Screen,
    pub selected_model_index: usize,
    pub sidebar_width: u16,
    pub is_loading_model: bool,
    pub is_generating: bool,
    pub notification: Option<NotificationState>,
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub tokens_per_second: f64,
    pub context_used: usize,
    pub context_limit: usize,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model_path: None,
            options: GenerateOptions::default(),
            messages: Vec::new(),
            current_screen: Screen::Chat,
            selected_model_index: 0,
            sidebar_width: 0,
            is_loading_model: false,
            is_generating: false,
            notification: None,
            tokens_generated: 0,
            prompt_tokens: 0,
            tokens_per_second: 0.0,
            context_used: 0,
            context_limit: 4096,
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
