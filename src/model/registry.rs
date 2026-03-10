use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::model::download::get_oxide_dir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub repo_id: String,
    pub filename: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub downloaded_at: DateTime<Utc>,
    pub quantization: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Registry {
    pub models: Vec<ModelEntry>,
}

impl Registry {
    fn path() -> Result<PathBuf> {
        Ok(get_oxide_dir()?.join("models.json"))
    }

    pub fn load() -> Result<Self> {
        let path = Self::path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path)?;
        let registry: Registry = serde_json::from_str(&content).context("Invalid registry file")?;
        Ok(registry)
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path()?;
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    pub fn add(&mut self, entry: ModelEntry) -> Result<()> {
        self.models.retain(|m| m.id != entry.id);
        self.models.push(entry);
        self.save()
    }

    pub fn remove(&mut self, id: &str) -> Result<Option<ModelEntry>> {
        let pos = self.models.iter().position(|m| m.id == id);
        if let Some(idx) = pos {
            let removed = self.models.remove(idx);
            self.save()?;
            Ok(Some(removed))
        } else {
            Ok(None)
        }
    }

    pub fn find_by_id(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn find_by_repo(&self, repo_id: &str) -> Vec<&ModelEntry> {
        self.models
            .iter()
            .filter(|m| m.repo_id == repo_id)
            .collect()
    }

    pub fn list(&self) -> &[ModelEntry] {
        &self.models
    }
}

pub fn generate_model_id(repo_id: &str, filename: &str) -> String {
    let repo_name = repo_id.split('/').last().unwrap_or(repo_id).to_lowercase();

    let quant = filename
        .to_uppercase()
        .split(".GGUF")
        .find(|s| s.starts_with("Q"))
        .map(|s| s.trim_end_matches('-').to_string());

    if let Some(q) = quant {
        format!("{}-{}", repo_name, q)
    } else {
        repo_name
    }
}

pub fn extract_quantization(filename: &str) -> Option<String> {
    filename
        .to_uppercase()
        .split(".GGUF")
        .find(|s| s.starts_with('Q'))
        .map(|s| s.trim_end_matches('-').to_string())
}

pub fn register_model(
    repo_id: &str,
    filename: &str,
    path: PathBuf,
    size_bytes: u64,
) -> Result<ModelEntry> {
    let mut registry = Registry::load()?;

    let id = generate_model_id(repo_id, filename);
    let quantization = extract_quantization(filename);

    let entry = ModelEntry {
        id: id.clone(),
        repo_id: repo_id.to_string(),
        filename: filename.to_string(),
        path,
        size_bytes,
        downloaded_at: Utc::now(),
        quantization,
    };

    registry.add(entry.clone())?;

    Ok(entry)
}

pub fn list_models() -> Result<Vec<ModelEntry>> {
    let registry = Registry::load()?;
    Ok(registry.list().to_vec())
}

pub fn unregister_model(id: &str) -> Result<Option<ModelEntry>> {
    let mut registry = Registry::load()?;
    registry.remove(id)
}

pub fn find_model(id: &str) -> Result<Option<ModelEntry>> {
    let registry = Registry::load()?;
    Ok(registry.find_by_id(id).cloned())
}

pub fn get_model_path(id: &str) -> Result<Option<PathBuf>> {
    let model = find_model(id)?;
    Ok(model.map(|m| m.path))
}
