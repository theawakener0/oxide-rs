//! Dynamic Batching for LLM Inference
//!
//! Groups incoming requests into small batches (max 8) that arrive within
//! a configurable time window (default 100ms) for improved throughput while
//! maintaining low latency.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

use crate::inference::Generator;

pub struct BatchConfig {
    pub max_batch_size: usize,
    pub batch_window_ms: u64,
    pub max_queue_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_window_ms: 100,
            max_queue_size: 100,
        }
    }
}

impl Clone for BatchConfig {
    fn clone(&self) -> Self {
        Self {
            max_batch_size: self.max_batch_size,
            batch_window_ms: self.batch_window_ms,
            max_queue_size: self.max_queue_size,
        }
    }
}

pub struct BatchRequest {
    pub id: u64,
    pub prompt: String,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub sender: oneshot::Sender<BatchResult>,
}

pub struct BatchResult {
    pub id: u64,
    pub result: Result<String, String>,
}

pub struct DynamicBatcher {
    config: BatchConfig,
    request_tx: mpsc::Sender<BatchRequest>,
    batch_counter: Arc<std::sync::atomic::AtomicU64>,
}

impl DynamicBatcher {
    pub fn new(config: BatchConfig) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.max_queue_size);
        let batch_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));

        let counter_clone = batch_counter.clone();
        let config_clone = config.clone();

        tokio::spawn(async move {
            Self::batcher_loop(request_rx, config_clone, counter_clone, None).await;
        });

        Self {
            config,
            request_tx,
            batch_counter,
        }
    }

    pub fn with_generator(config: BatchConfig, generator: Arc<tokio::sync::Mutex<Generator>>) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.max_queue_size);
        let batch_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));

        let counter_clone = batch_counter.clone();
        let config_clone = config.clone();

        tokio::spawn(async move {
            Self::batcher_loop(request_rx, config_clone, counter_clone, Some(generator)).await;
        });

        Self {
            config,
            request_tx,
            batch_counter,
        }
    }

    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    pub async fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<String, String> {
        let id = self
            .batch_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let (sender, receiver) = oneshot::channel();

        let request = BatchRequest {
            id,
            prompt,
            max_tokens,
            repeat_penalty,
            repeat_last_n,
            sender,
        };

        self.request_tx
            .send(request)
            .await
            .map_err(|_| "Batcher channel closed".to_string())?;

        match receiver.await {
            Ok(result) => result.result,
            Err(_) => Err("Request cancelled".to_string()),
        }
    }

    async fn batcher_loop(
        mut request_rx: mpsc::Receiver<BatchRequest>,
        config: BatchConfig,
        _counter: Arc<std::sync::atomic::AtomicU64>,
        generator: Option<Arc<tokio::sync::Mutex<Generator>>>,
    ) {
        let window_duration = Duration::from_millis(config.batch_window_ms);
        let max_batch_size = config.max_batch_size;

        let mut pending_requests: Vec<BatchRequest> = Vec::with_capacity(max_batch_size);
        let mut last_batch_time = Instant::now();

        loop {
            let time_since_last = last_batch_time.elapsed();

            if pending_requests.is_empty() {
                if let Some(req) = request_rx.recv().await {
                    pending_requests.push(req);
                    last_batch_time = Instant::now();
                } else {
                    break;
                }
            } else if time_since_last >= window_duration || pending_requests.len() >= max_batch_size {
                if !pending_requests.is_empty() {
                    let requests: Vec<_> = pending_requests.drain(..).collect();
                    Self::process_batch(requests, generator.clone()).await;
                    last_batch_time = Instant::now();
                }
            } else {
                match timeout(Duration::from_millis(1), request_rx.recv()).await {
                    Ok(Some(req)) => {
                        if pending_requests.len() < max_batch_size {
                            pending_requests.push(req);
                        } else {
                            if !pending_requests.is_empty() {
                                let requests: Vec<_> = pending_requests.drain(..).collect();
                                Self::process_batch(requests, generator.clone()).await;
                                last_batch_time = Instant::now();
                            }
                            pending_requests.push(req);
                        }
                    }
                    Ok(None) => {
                        if !pending_requests.is_empty() {
                            let requests: Vec<_> = pending_requests.drain(..).collect();
                            Self::process_batch(requests, generator.clone()).await;
                        }
                        break;
                    }
                    Err(_) => {
                        if !pending_requests.is_empty() {
                            let requests: Vec<_> = pending_requests.drain(..).collect();
                            Self::process_batch(requests, generator.clone()).await;
                            last_batch_time = Instant::now();
                        }
                    }
                }
            }
        }
    }

    async fn process_batch(requests: Vec<BatchRequest>, generator: Option<Arc<tokio::sync::Mutex<Generator>>>) {
        if requests.is_empty() {
            return;
        }

        tracing::debug!("Processing batch of {} requests", requests.len());

        match generator {
            Some(gen) => {
                let prompts: Vec<String> = requests.iter().map(|r| r.prompt.clone()).collect();
                let max_tokens = requests.first().map(|r| r.max_tokens).unwrap_or(512);
                let repeat_penalty = requests.first().map(|r| r.repeat_penalty).unwrap_or(1.1);
                let repeat_last_n = requests.first().map(|r| r.repeat_last_n).unwrap_or(64);

                let results = tokio::task::spawn_blocking(move || {
                    let mut gen = gen.blocking_lock();
                    gen.generate_batch(prompts, max_tokens, repeat_penalty, repeat_last_n)
                })
                .await;

                match results {
                    Ok(Ok(outputs)) => {
                        for (req, result) in requests.into_iter().zip(outputs.into_iter()) {
                            let _ = req.sender.send(BatchResult {
                                id: req.id,
                                result: Ok(result),
                            });
                        }
                    }
                    Ok(Err(e)) => {
                        for req in requests {
                            let _ = req.sender.send(BatchResult {
                                id: req.id,
                                result: Err(e.to_string()),
                            });
                        }
                    }
                    Err(e) => {
                        for req in requests {
                            let _ = req.sender.send(BatchResult {
                                id: req.id,
                                result: Err(format!("Task join error: {}", e)),
                            });
                        }
                    }
                }
            }
            None => {
                for req in requests {
                    let _ = req.sender.send(BatchResult {
                        id: req.id,
                        result: Err("Generator not connected to batcher".to_string()),
                    });
                }
            }
        }
    }
}

impl Clone for DynamicBatcher {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            request_tx: self.request_tx.clone(),
            batch_counter: self.batch_counter.clone(),
        }
    }
}

pub struct DynamicBatcherHandle {
    batcher: DynamicBatcher,
}

impl DynamicBatcherHandle {
    pub fn new(config: BatchConfig) -> Self {
        Self {
            batcher: DynamicBatcher::new(config),
        }
    }

    pub fn with_generator(config: BatchConfig, generator: Arc<tokio::sync::Mutex<Generator>>) -> Self {
        Self {
            batcher: DynamicBatcher::with_generator(config, generator),
        }
    }

    pub async fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<String, String> {
        self.batcher
            .generate(prompt, max_tokens, repeat_penalty, repeat_last_n)
            .await
    }
}

impl Clone for DynamicBatcherHandle {
    fn clone(&self) -> Self {
        Self {
            batcher: self.batcher.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.batch_window_ms, 100);
        assert_eq!(config.max_queue_size, 100);
    }
}
