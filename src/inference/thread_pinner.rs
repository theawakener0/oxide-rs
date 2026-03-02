//! Thread Pinning for CPU Inference
//!
//! Pins inference threads to CPU cores for consistent performance
//! and reduced context switching overhead.
//!
//! This implementation uses a portable approach that works on Linux/macOS/Windows.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::OnceLock;

pub struct ThreadPinnerConfig {
    pub num_threads: usize,
    pub reserve_cores: usize,
    pub enabled: bool,
}

impl Default for ThreadPinnerConfig {
    fn default() -> Self {
        let total_cores = num_cpus::get();
        let num_threads = total_cores.saturating_sub(1).max(1);

        Self {
            num_threads,
            reserve_cores: 0,
            enabled: true,
        }
    }
}

impl Clone for ThreadPinnerConfig {
    fn clone(&self) -> Self {
        Self {
            num_threads: self.num_threads,
            reserve_cores: self.reserve_cores,
            enabled: self.enabled,
        }
    }
}

impl ThreadPinnerConfig {
    pub fn new(num_threads: usize, reserve_cores: usize) -> Self {
        Self {
            num_threads,
            reserve_cores,
            enabled: true,
        }
    }

    pub fn auto(total_cores: usize) -> Self {
        let num_threads = total_cores.saturating_sub(1).max(1);
        Self {
            num_threads,
            reserve_cores: 0,
            enabled: true,
        }
    }
}

static THREAD_PINNER: OnceLock<ThreadPinner> = OnceLock::new();

#[derive(Clone)]
pub struct ThreadPinner {
    config: ThreadPinnerConfig,
    core_ids: Vec<usize>,
}

impl ThreadPinner {
    pub fn init(config: ThreadPinnerConfig) -> &'static Self {
        THREAD_PINNER.get_or_init(|| {
            let core_ids = Self::get_available_cores(config.reserve_cores, config.num_threads);

            tracing::info!(
                "Thread pinning initialized: {} threads on cores {:?}",
                config.num_threads,
                core_ids
            );

            Self { config, core_ids }
        })
    }

    pub fn get() -> &'static Self {
        THREAD_PINNER.get_or_init(|| {
            let config = ThreadPinnerConfig::default();
            let core_ids = Self::get_available_cores(config.reserve_cores, config.num_threads);

            Self { config, core_ids }
        })
    }

    fn get_available_cores(reserve_cores: usize, num_threads: usize) -> Vec<usize> {
        let total_cores = num_cpus::get();

        let start = reserve_cores.min(total_cores);
        let available: Vec<usize> = (start..total_cores).take(num_threads).collect();

        if available.len() < num_threads {
            tracing::warn!(
                "Requested {} threads but only {} cores available",
                num_threads,
                available.len()
            );
        }

        available
    }

    #[cfg(target_os = "linux")]
    pub fn pin_current_thread(&self) -> bool {
        if !self.config.enabled || self.core_ids.is_empty() {
            return false;
        }

        if let Some(core_id) = self.core_ids.first() {
            let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            unsafe {
                libc::CPU_SET(*core_id, &mut cpuset);
            }

            let result = unsafe {
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset)
            };

            if result == 0 {
                tracing::debug!("Pinned thread to core {}", core_id);
                return true;
            } else {
                tracing::warn!("Failed to pin thread to core {}: {}", core_id, result);
                return false;
            }
        }
        false
    }

    #[cfg(target_os = "linux")]
    pub fn pin_thread_by_index(&self, thread_index: usize) -> bool {
        if !self.config.enabled || self.core_ids.is_empty() {
            return false;
        }

        let core_id = self.core_ids[thread_index % self.core_ids.len()];

        let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::CPU_SET(core_id, &mut cpuset);
        }

        let result =
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset) };

        if result == 0 {
            tracing::debug!("Pinned thread {} to core {}", thread_index, core_id);
            return true;
        } else {
            tracing::warn!(
                "Failed to pin thread {} to core {}: {}",
                thread_index,
                core_id,
                result
            );
            return false;
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn pin_current_thread(&self) -> bool {
        if !self.config.enabled || self.core_ids.is_empty() {
            return false;
        }

        tracing::warn!("Thread pinning not fully supported on this platform");
        false
    }

    #[cfg(not(target_os = "linux"))]
    pub fn pin_thread_by_index(&self, _thread_index: usize) -> bool {
        if !self.config.enabled || self.core_ids.is_empty() {
            return false;
        }

        tracing::warn!("Thread pinning not fully supported on this platform");
        false
    }

    pub fn build_thread_pool(&self) -> Result<ThreadPool, Box<dyn std::error::Error>> {
        let core_ids = self.core_ids.clone();
        let enabled = self.config.enabled;

        let pool = ThreadPoolBuilder::new()
            .num_threads(self.core_ids.len())
            .spawn_handler(move |thread| {
                let index = thread.index();
                if enabled {
                    let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                    let core_id = core_ids[index % core_ids.len()];
                    unsafe {
                        libc::CPU_SET(core_id, &mut cpuset);
                    }
                    unsafe {
                        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
                    }
                }
                thread.run();
                Ok(())
            })
            .build()?;

        Ok(pool)
    }

    pub fn num_threads(&self) -> usize {
        self.core_ids.len()
    }

    pub fn core_ids(&self) -> &[usize] {
        &self.core_ids
    }
}

pub fn pin_threads_to_cores(num_threads: usize, reserve_cores: usize) -> ThreadPinner {
    let config = ThreadPinnerConfig::new(num_threads, reserve_cores);
    ThreadPinner::init(config).clone()
}

pub fn init_thread_pinner(config: ThreadPinnerConfig) -> &'static ThreadPinner {
    ThreadPinner::init(config)
}

pub fn get_thread_pinner() -> &'static ThreadPinner {
    ThreadPinner::get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pinner_config_defaults() {
        let config = ThreadPinnerConfig::default();
        assert!(config.num_threads > 0);
    }

    #[test]
    fn test_thread_pinner_auto() {
        let config = ThreadPinnerConfig::auto(4);
        assert_eq!(config.num_threads, 3);
    }
}
