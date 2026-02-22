use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct History {
    messages: Vec<Message>,
    #[serde(skip)]
    path: PathBuf,
}

impl History {
    pub fn load() -> Self {
        let path = Self::get_history_path();

        if path.exists() {
            match fs::read_to_string(&path) {
                Ok(content) => match serde_json::from_str(&content) {
                    Ok(history) => {
                        tracing::debug!("Loaded history from {:?}", path);
                        return history;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse history: {}", e);
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read history: {}", e);
                }
            }
        }

        Self {
            messages: Vec::new(),
            path,
        }
    }

    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            path: Self::get_history_path(),
        }
    }

    fn get_history_path() -> PathBuf {
        let mut path = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push("oxide");
        if !path.exists() {
            fs::create_dir_all(&path).ok();
        }
        path.push("history.json");
        path
    }

    pub fn add(&mut self, role: &str, content: &str) {
        self.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn save(&self) {
        if self.messages.is_empty() {
            return;
        }

        let content = match serde_json::to_string_pretty(&self) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to serialize history: {}", e);
                return;
            }
        };

        if let Err(e) = fs::write(&self.path, content) {
            tracing::warn!("Failed to save history: {}", e);
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.save();
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn last_n_messages(&self, n: usize) -> &[Message] {
        if self.messages.len() <= n {
            &self.messages
        } else {
            &self.messages[self.messages.len() - n..]
        }
    }

    pub fn format_for_context(&self, max_messages: usize) -> String {
        let messages = self.last_n_messages(max_messages);
        messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}
