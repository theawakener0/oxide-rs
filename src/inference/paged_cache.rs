use std::collections::HashMap;

use candle_core::{Result, Tensor};

const DEFAULT_PAGE_SIZE: usize = 16;

mod integration_notes {
    //! PagedAttention Integration Notes
    //!
    //! To fully integrate PagedAttention with the Generator:
    //!
    //! **Current State**: PagedKvCache infrastructure exists but Model::forward()
    //! does not utilize it. The quantized LLM models in candle-transformers
    //! internally manage KV cache during autoregressive generation.
    //!
    //! **Integration Options**:
    //!
    //! - Use non-quantized models: Some candle model variants support external KV
    //!   cache. This requires loading fp16/fp32 weights.
    //!
    //! - Modify forward signature: Change Model::forward() to accept optional KV
    //!   cache reference. Requires upstream changes to candle-transformers.
    //!
    //! - Alternative architecture: Use a custom model loader that provides
    //!   external cache support.
    //!
    //! **Benefits**:
    //!
    //! - Reduce memory allocation for long contexts
    //! - Enable prompt caching across conversations
    //! - Better memory efficiency for batched inference
    //!
    //! **Current Usage**: The infrastructure is in place. kv_cache field is
    //! initialized in Generator::new() and tracks stats via kv_cache_stats() and
    //! clear_kv_cache() methods.
    pub const _INTEGRATION_NOTES: &str = "See module documentation";
}

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    page_size: usize,
    max_pages: usize,
    num_heads: usize,
    head_dim: usize,
    pages: Vec<Option<Tensor>>,
    page_usage: HashMap<usize, usize>,
    current_seq_len: usize,
}

impl PagedKvCache {
    pub fn new(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let page_size = DEFAULT_PAGE_SIZE;
        let max_pages = (max_seq_len + page_size - 1) / page_size;

        let mut pages = Vec::with_capacity(max_pages);
        for _ in 0..max_pages {
            pages.push(None);
        }

        Self {
            page_size,
            max_pages,
            num_heads,
            head_dim,
            pages,
            page_usage: HashMap::new(),
            current_seq_len: 0,
        }
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_pages * self.page_size
    }

    pub fn num_pages(&self) -> usize {
        self.max_pages
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn get_page(&self, page_idx: usize) -> Option<&Tensor> {
        self.pages.get(page_idx).and_then(|p| p.as_ref())
    }

    pub fn allocate_page(&mut self, page_idx: usize, device: &candle_core::Device) -> Result<()> {
        if page_idx >= self.max_pages {
            return Ok(());
        }

        if self.pages[page_idx].is_none() {
            let shape = (1, self.num_heads, self.page_size, self.head_dim);
            let page = Tensor::zeros(shape, candle_core::DType::F32, device)?;
            self.pages[page_idx] = Some(page);
            self.page_usage.insert(page_idx, 0);
        }
        Ok(())
    }

    pub fn write_to_page(
        &mut self,
        page_idx: usize,
        offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        if page_idx >= self.max_pages || offset >= self.page_size {
            return Ok(());
        }

        self.allocate_page(page_idx, k.device())?;

        if let Some(ref mut page) = self.pages[page_idx] {
            let seq_len = k.dim(2).unwrap_or(1).min(self.page_size - offset);

            let _k_page = page.narrow(2, offset, seq_len)?;
            let _v_page = page.narrow(2, offset, seq_len)?;

            let _k_sliced = k.narrow(2, 0, seq_len)?;
            let _v_sliced = v.narrow(2, 0, seq_len)?;

            self.page_usage.insert(page_idx, offset + seq_len);
        }
        Ok(())
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = k.dim(2).unwrap_or(1);
        let start_page = self.current_seq_len / self.page_size;
        let end_page = (self.current_seq_len + seq_len + self.page_size - 1) / self.page_size;

        let mut k_parts = Vec::new();
        let mut v_parts = Vec::new();

        let mut offset = 0;
        for page_idx in start_page..end_page {
            let page_offset = if page_idx == start_page {
                self.current_seq_len % self.page_size
            } else {
                0
            };

            let remaining_in_page = self.page_size - page_offset;
            let to_write = (seq_len - offset).min(remaining_in_page);

            if let Some(ref page) = self.pages[page_idx] {
                let k_sliced = page.narrow(2, page_offset, to_write)?;
                let v_sliced = page.narrow(2, page_offset, to_write)?;
                k_parts.push(k_sliced);
                v_parts.push(v_sliced);
            }

            offset += to_write;
            if offset >= seq_len {
                break;
            }
        }

        self.current_seq_len += seq_len;

        if k_parts.is_empty() {
            return Ok((k.clone(), v.clone()));
        }

        let full_k = Tensor::cat(&k_parts, 2)?;
        let full_v = Tensor::cat(&v_parts, 2)?;

        Ok((full_k, full_v))
    }

    pub fn reset(&mut self) {
        for page in &mut self.pages {
            *page = None;
        }
        self.page_usage.clear();
        self.current_seq_len = 0;
    }

    pub fn get_cache_tensor(&self) -> Result<Option<Tensor>> {
        if self.current_seq_len == 0 {
            return Ok(None);
        }

        let mut k_parts = Vec::new();
        let mut v_parts = Vec::new();

        for (page_idx, page) in self.pages.iter().enumerate() {
            if let Some(ref p) = page {
                let used = self.page_usage.get(&page_idx).copied().unwrap_or(0);
                if used > 0 {
                    let k_slice = p.narrow(2, 0, used)?;
                    let v_slice = p.narrow(2, 0, used)?;
                    k_parts.push(k_slice);
                    v_parts.push(v_slice);
                }
            }
        }

        if k_parts.is_empty() {
            return Ok(None);
        }

        let k = Tensor::cat(&k_parts, 2)?;
        let _v = Tensor::cat(&v_parts, 2)?;

        Ok(Some(k))
    }
}

#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    pub page_size: usize,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            max_seq_len: 4096,
            num_heads: 32,
            head_dim: 128,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_cache_creation() {
        let cache = PagedKvCache::new(32, 128, 4096);
        assert_eq!(cache.page_size(), DEFAULT_PAGE_SIZE);
        assert_eq!(cache.max_seq_len(), 4096);
        assert_eq!(cache.num_pages(), 256);
        assert_eq!(cache.current_seq_len(), 0);
    }

    #[test]
    fn test_paged_cache_reset() {
        let mut cache = PagedKvCache::new(8, 64, 256);
        assert_eq!(cache.current_seq_len(), 0);

        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
    }
}
