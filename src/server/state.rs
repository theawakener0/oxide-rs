use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::RwLock;
use std::sync::Mutex;

use crate::inference::Generator;
use crate::GenerateOptions;

pub struct AppState {
    model_cache: RwLock<HashMap<String, Arc<Mutex<Generator>>>>,
    default_options: GenerateOptions,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model_cache: RwLock::new(HashMap::new()),
            default_options: GenerateOptions::default(),
        }
    }

    pub async fn get_or_load_model(
        &self,
        model_path: &str,
    ) -> Result<Arc<Mutex<Generator>>, Box<dyn std::error::Error + Send + Sync>> {
        {
            let cache = self.model_cache.read().await;
            if let Some(generator) = cache.get(model_path) {
                let gen = generator.lock().unwrap();
                let metadata = gen.metadata();
                tracing::info!(
                    "[MODEL] Using cached model: {} | layers: {} | ctx: {}",
                    model_path,
                    metadata.n_layer,
                    metadata.context_length
                );
                return Ok(generator.clone());
            }
        }

        tracing::info!("[MODEL] Loading model: {}", model_path);

        let path = Path::new(model_path);
        if !path.exists() {
            return Err(format!("Model file not found: {}", model_path).into());
        }

        let load_start = std::time::Instant::now();
        
        let generator = Generator::new(
            &path.to_path_buf(),
            None,
            self.default_options.temperature,
            self.default_options.top_p,
            self.default_options.top_k,
            self.default_options.seed,
            self.default_options.system_prompt.clone(),
            self.default_options.batch_size,
        )?;

        let load_time = load_start.elapsed();
        let metadata = generator.metadata();

        tracing::info!(
            "[MODEL] Model loaded successfully: {} | quant: {:?} | layers: {} | embed: {} | ctx: {} | vocab: {} | loaded in {:.2}s",
            metadata.name,
            metadata.quantization,
            metadata.n_layer,
            metadata.n_embd,
            metadata.context_length,
            metadata.vocab_size,
            load_time.as_secs_f32()
        );

        let generator = Arc::new(Mutex::new(generator));

        {
            let mut cache = self.model_cache.write().await;
            cache.insert(model_path.to_string(), generator.clone());
        }

        Ok(generator)
    }

    pub async fn list_models(&self) -> Vec<String> {
        let cache = self.model_cache.read().await;
        cache.keys().cloned().collect()
    }

    pub fn set_default_options(&mut self, options: GenerateOptions) {
        self.default_options = options;
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
