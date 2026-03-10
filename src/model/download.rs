use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use hf_hub::Repo;

#[derive(Debug, Clone)]
pub struct RepoFile {
    pub name: String,
    pub size: u64,
    pub rfilename: String,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub repo_id: String,
    pub files: Vec<RepoFile>,
    pub total_size: u64,
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub filename: String,
}

pub fn get_hf_cache_dir() -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .context("Could not find cache directory")?
        .join("huggingface")
        .join("hub");
    Ok(cache_dir)
}

pub fn get_oxide_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not find home directory")?;
    let oxide_dir = home.join(".oxide");
    std::fs::create_dir_all(&oxide_dir)?;
    Ok(oxide_dir)
}

pub fn list_repo_files(repo_id: &str) -> Result<Vec<RepoFile>> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);

    let client = reqwest::blocking::Client::new();
    let response = client.get(&url).send()?.json::<serde_json::Value>()?;

    let files: Vec<RepoFile> = if let Some(items) = response.as_array() {
        items
            .iter()
            .filter_map(|item| {
                let type_ = item.get("type")?.as_str()?;
                if type_ != "file" {
                    return None;
                }
                let size = item.get("size")?.as_u64()?;
                let path = item.get("path")?.as_str()?;
                Some(RepoFile {
                    name: path.to_string(),
                    size,
                    rfilename: path.to_string(),
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(files)
}

pub fn get_model_info(repo_id: &str) -> Result<ModelInfo> {
    let files = list_repo_files(repo_id)?;
    let total_size: u64 = files.iter().map(|f| f.size).sum();

    Ok(ModelInfo {
        repo_id: repo_id.to_string(),
        files,
        total_size,
    })
}

pub fn find_gguf_file(files: &[RepoFile]) -> Option<&RepoFile> {
    let q4_priority = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q4"];
    let q5_priority = ["Q5_K_S", "Q5_K_M", "Q5_0", "Q5"];
    let q6_priority = ["Q6_K", "Q6"];
    let q8_priority = ["Q8_0", "Q8"];

    for prefix in q4_priority.iter() {
        if let Some(f) = files
            .iter()
            .find(|f| f.rfilename.to_uppercase().contains(prefix) && f.rfilename.ends_with(".gguf"))
        {
            return Some(f);
        }
    }

    for prefix in q5_priority.iter() {
        if let Some(f) = files
            .iter()
            .find(|f| f.rfilename.to_uppercase().contains(prefix) && f.rfilename.ends_with(".gguf"))
        {
            return Some(f);
        }
    }

    for prefix in q6_priority.iter() {
        if let Some(f) = files
            .iter()
            .find(|f| f.rfilename.to_uppercase().contains(prefix) && f.rfilename.ends_with(".gguf"))
        {
            return Some(f);
        }
    }

    for prefix in q8_priority.iter() {
        if let Some(f) = files
            .iter()
            .find(|f| f.rfilename.to_uppercase().contains(prefix) && f.rfilename.ends_with(".gguf"))
        {
            return Some(f);
        }
    }

    files.iter().find(|f| f.rfilename.ends_with(".gguf"))
}

pub fn is_model_cached(repo_id: &str, filename: &str) -> Result<bool> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));

    let cache_path = repo.get(filename)?;
    Ok(cache_path.exists())
}

pub fn get_cached_path(repo_id: &str, filename: &str) -> Result<Option<PathBuf>> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));

    let path = repo.get(filename)?;
    if path.exists() {
        Ok(Some(path))
    } else {
        Ok(None)
    }
}

pub fn download_model<F>(
    repo_id: &str,
    filename: Option<&str>,
    mut progress_callback: F,
) -> Result<(PathBuf, String)>
where
    F: FnMut(DownloadProgress),
{
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));

    let files = list_repo_files(repo_id)?;

    let target_file = if let Some(name) = filename {
        if name.ends_with(".gguf") {
            name.to_string()
        } else {
            format!("{}.gguf", name)
        }
    } else {
        let found = find_gguf_file(&files).context("No GGUF file found in repository")?;
        found.rfilename.clone()
    };

    let file_info = files
        .iter()
        .find(|f| f.rfilename == target_file)
        .with_context(|| format!("File {} not found in repository", target_file))?;

    if let Ok(Some(cached)) = get_cached_path(repo_id, &target_file) {
        return Ok((cached, target_file));
    }

    progress_callback(DownloadProgress {
        bytes_downloaded: 0,
        total_bytes: file_info.size,
        filename: target_file.clone(),
    });

    let path = repo.get(&target_file)?;

    progress_callback(DownloadProgress {
        bytes_downloaded: file_info.size,
        total_bytes: file_info.size,
        filename: target_file.clone(),
    });

    Ok((path, target_file))
}

pub fn parse_repo_id(input: &str) -> (String, Option<String>) {
    if input.contains('/') && input.ends_with(".gguf") {
        let parts: Vec<&str> = input.split('/').collect();
        let repo_id = parts[..parts.len() - 1].join("/");
        let filename = parts.last().unwrap().to_string();
        (repo_id, Some(filename))
    } else if input.ends_with(".gguf") {
        let filename = input.to_string();
        let repo_id = "".to_string();
        (repo_id, Some(filename))
    } else {
        (input.to_string(), None)
    }
}

pub fn format_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if size >= GB {
        format!("{:.1}GB", size as f64 / GB as f64)
    } else if size >= MB {
        format!("{:.1}MB", size as f64 / MB as f64)
    } else if size >= KB {
        format!("{:.1}KB", size as f64 / KB as f64)
    } else {
        format!("{}B", size)
    }
}
