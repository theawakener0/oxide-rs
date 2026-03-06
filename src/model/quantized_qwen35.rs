use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, Activation, Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::utils::repeat_kv;

#[derive(Debug, Clone)]
struct ZeroCenteredRmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl ZeroCenteredRmsNorm {
    fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: (weight.dequantize(&weight.device())? + 1f64)?,
            eps,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        })
    }
}

impl Module for ZeroCenteredRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

#[derive(Debug, Clone)]
struct GatedRmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl GatedRmsNorm {
    fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: weight.dequantize(&weight.device())?,
            eps,
            span: tracing::span!(tracing::Level::TRACE, "gated-rms-norm"),
        })
    }

    fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let x = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        x.broadcast_mul(&candle_nn::ops::silu(gate)?)
    }
}

#[derive(Debug)]
struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    fn dequantized(&mut self, name: &str) -> Result<Tensor> {
        self.tensor(name)?.dequantize(&self.device)
    }

    fn tensor_dims(&self, name: &str) -> Result<Vec<usize>> {
        match self.ct.tensor_infos.get(name) {
            Some(info) => Ok(info.shape.dims().to_vec()),
            None => candle_core::bail!("cannot find tensor info for {name}"),
        }
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        rotary_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim,
        })
    }

    fn apply(&self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, head_dim) = xs.dims4()?;
        let cos = self
            .cos
            .narrow(0, offset, seq_len)?
            .to_dtype(xs.dtype())?
            .contiguous()?;
        let sin = self
            .sin
            .narrow(0, offset, seq_len)?
            .to_dtype(xs.dtype())?
            .contiguous()?;
        let xs_rot = xs.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
        let xs_pass = xs.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, &cos, &sin)?;
        Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()
    }
}

fn softplus(xs: &Tensor) -> Result<Tensor> {
    (xs.exp()? + 1f64)?.log()
}

fn l2_norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    xs.broadcast_div(&((xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?))
}

fn repeat_heads(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (b, l, h, d) = xs.dims4()?;
    let refs: Vec<&Tensor> = std::iter::repeat(xs).take(n_rep).collect();
    Tensor::cat(&refs, 2)?.reshape((b, l, h * n_rep, d))
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_proj: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
            up_proj: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
            down_proj: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
            act_fn: Activation::Silu,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: ZeroCenteredRmsNorm,
    k_norm: ZeroCenteredRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
    span: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: gg.qmatmul(&format!("{prefix}.attn_q.weight"))?,
            k_proj: gg.qmatmul(&format!("{prefix}.attn_k.weight"))?,
            v_proj: gg.qmatmul(&format!("{prefix}.attn_v.weight"))?,
            o_proj: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
            q_norm: ZeroCenteredRmsNorm::from_qtensor(
                gg.tensor(&format!("{prefix}.attn_q_norm.weight"))?,
                rms_norm_eps,
            )?,
            k_norm: ZeroCenteredRmsNorm::from_qtensor(
                gg.tensor(&format!("{prefix}.attn_k_norm.weight"))?,
                rms_norm_eps,
            )?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: ConcatKvCache::new(2),
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l, _) = x.dims3()?;
        let q_full = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q_full = q_full.reshape((b, l, self.num_heads, self.head_dim * 2))?;
        let q = q_full.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_full
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b, l, self.num_heads * self.head_dim))?;

        let q = self.q_norm.forward(&q.transpose(1, 2)?.contiguous()?)?;
        let k = self.k_norm.forward(
            &k.reshape((b, l, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?,
        )?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.rotary_emb.apply(&q, offset)?;
        let k = self.rotary_emb.apply(&k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let probs = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?)? * scale)?)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        let ctx = ctx.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct SsmWeights {
    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_b: QMatMul,
    in_proj_a: QMatMul,
    ssm_a: Tensor,
    dt_bias: Tensor,
    conv_weight: Tensor,
    norm: GatedRmsNorm,
    out_proj: QMatMul,
    conv_kernel: usize,
    key_dim: usize,
    value_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    span: tracing::Span,
}

impl SsmWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        rms_norm_eps: f64,
        num_k_heads: usize,
        prefix: &str,
    ) -> Result<Self> {
        let md = gg.metadata();
        let qkv_dims = gg.tensor_dims(&format!("{prefix}.attn_qkv.weight"))?;
        let conv_dims = gg.tensor_dims(&format!("{prefix}.ssm_conv1d.weight"))?;
        let a_dims = gg.tensor_dims(&format!("{prefix}.ssm_a"))?;

        let value_dim = md
            .get("qwen35.ssm.inner_size")
            .ok_or_else(|| {
                candle_core::Error::Msg("cannot find qwen35.ssm.inner_size in metadata".to_string())
            })?
            .to_u64()? as usize;
        let conv_dim = *qkv_dims
            .iter()
            .max()
            .ok_or_else(|| candle_core::Error::Msg("invalid attn_qkv weight shape".to_string()))?;
        let key_dim = (conv_dim - value_dim) / 2;
        let num_v_heads = a_dims[0];
        let head_v_dim = value_dim / num_v_heads;
        let head_k_dim = key_dim / num_k_heads;
        let conv_kernel = *conv_dims.iter().min().ok_or_else(|| {
            candle_core::Error::Msg("invalid ssm_conv1d weight shape".to_string())
        })?;

        let conv_weight = gg.dequantized(&format!("{prefix}.ssm_conv1d.weight"))?;
        let conv_weight = if conv_weight.dims2()? == (conv_kernel, conv_dim) {
            conv_weight.t()?.contiguous()?
        } else {
            conv_weight.contiguous()?
        };

        Ok(Self {
            in_proj_qkv: gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?,
            in_proj_z: gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?,
            in_proj_b: gg.qmatmul(&format!("{prefix}.ssm_beta.weight"))?,
            in_proj_a: gg.qmatmul(&format!("{prefix}.ssm_alpha.weight"))?,
            ssm_a: gg
                .dequantized(&format!("{prefix}.ssm_a"))?
                .reshape((1, 1, num_v_heads))?,
            dt_bias: gg.dequantized(&format!("{prefix}.ssm_dt.bias"))?.reshape((
                1,
                1,
                num_v_heads,
            ))?,
            conv_weight,
            norm: GatedRmsNorm::from_qtensor(
                gg.tensor(&format!("{prefix}.ssm_norm.weight"))?,
                rms_norm_eps,
            )?,
            out_proj: gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?,
            conv_kernel,
            key_dim,
            value_dim,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_state: None,
            recurrent_state: None,
            span: tracing::span!(tracing::Level::TRACE, "ssm"),
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l, _) = x.dims3()?;
        if b != 1 || l != 1 {
            candle_core::bail!(
                "qwen35 SSM block expects single-token inputs, got {:?}",
                x.dims()
            )
        }

        let mixed_qkv = self.in_proj_qkv.forward(x)?.transpose(1, 2)?;
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape((b, l, self.num_v_heads, self.head_v_dim))?;
        let beta = candle_nn::ops::sigmoid(&self.in_proj_b.forward(x)?)?;
        let a = self.in_proj_a.forward(x)?;

        let current = mixed_qkv.contiguous()?;
        let conv_out = if let Some(prev_state) = &self.conv_state {
            let full = Tensor::cat(&[prev_state, &current], 2)?;
            let weight = self.conv_weight.unsqueeze(0)?;
            let out = full.broadcast_mul(&weight)?.sum_keepdim(2)?;
            self.conv_state = Some(full.narrow(2, 1, self.conv_kernel - 1)?);
            out
        } else {
            let left = Tensor::zeros(
                (b, self.key_dim * 2 + self.value_dim, self.conv_kernel - 1),
                current.dtype(),
                current.device(),
            )?;
            let full = Tensor::cat(&[&left, &current], 2)?;
            let weight = self.conv_weight.unsqueeze(0)?;
            let out = full.broadcast_mul(&weight)?.sum_keepdim(2)?;
            self.conv_state = Some(full.narrow(2, 1, self.conv_kernel - 1)?);
            out
        };
        let mixed_qkv = candle_nn::ops::silu(&conv_out)?.transpose(1, 2)?;

        let query = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?.reshape((
            b,
            l,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let key = mixed_qkv
            .narrow(D::Minus1, self.key_dim, self.key_dim)?
            .reshape((b, l, self.num_k_heads, self.head_k_dim))?;
        let value = mixed_qkv
            .narrow(D::Minus1, self.key_dim * 2, self.value_dim)?
            .reshape((b, l, self.num_v_heads, self.head_v_dim))?;

        let query = l2_norm(&query, 1e-6)?;
        let key = l2_norm(&key, 1e-6)?;
        let rep = self.num_v_heads / self.num_k_heads;
        let query = repeat_heads(&query, rep)?;
        let key = repeat_heads(&key, rep)?;
        let g_base = self.ssm_a.exp()?.neg()?;
        let g = g_base.broadcast_mul(&softplus(&(a.to_dtype(DType::F32)? + &self.dt_bias)?)?)?;

        let query = query.squeeze(1)?;
        let key = key.squeeze(1)?;
        let value = value.squeeze(1)?;
        let beta = beta.squeeze(1)?;
        let g = g.squeeze(1)?.exp()?;

        let mut state = match &self.recurrent_state {
            Some(s) => s.broadcast_mul(&g.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?,
            None => Tensor::zeros(
                (b, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                x.device(),
            )?,
        };
        let kv_mem = state
            .broadcast_mul(&key.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?;
        let delta = (value.to_dtype(DType::F32)?.broadcast_sub(&kv_mem)?)
            .broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        state = state.broadcast_add(
            &key.unsqueeze(D::Minus1)?
                .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?,
        )?;
        self.recurrent_state = Some(state.clone());

        let out = state
            .broadcast_mul(&query.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?;
        let out = out.reshape((b * self.num_v_heads, self.head_v_dim))?;
        let gate = z.reshape((b * self.num_v_heads, self.head_v_dim))?;
        let out = self.norm.forward(
            &out.to_dtype(x.dtype())?,
            &gate.reshape((b * self.num_v_heads, self.head_v_dim))?,
        )?;
        let out = out.reshape((b, l, self.value_dim))?;
        self.out_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

#[derive(Debug, Clone)]
enum Mixer {
    Attention(AttentionWeights),
    Ssm(SsmWeights),
}

impl Mixer {
    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        match self {
            Self::Attention(attn) => attn.forward(x, offset),
            Self::Ssm(ssm) => ssm.forward(x),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Self::Attention(attn) => attn.clear_kv_cache(),
            Self::Ssm(ssm) => ssm.clear_kv_cache(),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    mixer: Mixer,
    mlp: MlpWeights,
    ln1: ZeroCenteredRmsNorm,
    ln2: ZeroCenteredRmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        full_attention_interval: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        num_linear_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let mixer = if (layer_idx + 1) % full_attention_interval == 0 {
            Mixer::Attention(AttentionWeights::new(
                gg,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                rms_norm_eps,
                rotary,
                &prefix,
            )?)
        } else {
            Mixer::Ssm(SsmWeights::new(
                gg,
                rms_norm_eps,
                num_linear_heads,
                &prefix,
            )?)
        };
        Ok(Self {
            mixer,
            mlp: MlpWeights::new(gg, &prefix)?,
            ln1: ZeroCenteredRmsNorm::from_qtensor(
                gg.tensor(&format!("{prefix}.attn_norm.weight"))?,
                rms_norm_eps,
            )?,
            ln2: ZeroCenteredRmsNorm::from_qtensor(
                gg.tensor(&format!("{prefix}.post_attention_norm.weight"))?,
                rms_norm_eps,
            )?,
        })
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.mixer.forward(&h, offset)?;
        let x = x.broadcast_add(&h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x.broadcast_add(&h2)
    }

    fn clear_kv_cache(&mut self) {
        self.mixer.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: ZeroCenteredRmsNorm,
    lm_head: QMatMul,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen35.attention.head_count")?.to_u64()? as usize;
        let num_kv_heads = md_get("qwen35.attention.head_count_kv")?.to_u64()? as usize;
        let num_layers = md_get("qwen35.block_count")?.to_u64()? as usize;
        let hidden_size = md_get("qwen35.embedding_length")?.to_u64()? as usize;
        let head_dim = md_get("qwen35.attention.key_length")?.to_u64()? as usize;
        let max_position_embeddings = md_get("qwen35.context_length")?.to_u64()? as usize;
        let rms_norm_eps = md_get("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen35.rope.freq_base")?.to_f32()? as f64;
        let full_attention_interval = md_get("qwen35.full_attention_interval")?.to_u64()? as usize;
        let num_linear_heads = md_get("qwen35.ssm.group_count")?.to_u64()? as usize;

        let rotary_dim = gg
            .metadata()
            .get("qwen35.rope.dimension_sections")
            .and_then(|v| v.to_vec().ok())
            .map(|vals| vals.iter().filter_map(|v| v.to_u64().ok()).sum::<u64>() as usize * 2)
            .unwrap_or(head_dim / 4);

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);
        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            rotary_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                i,
                full_attention_interval,
                num_attention_heads,
                num_kv_heads,
                num_linear_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
            )?);
        }

        let norm =
            ZeroCenteredRmsNorm::from_qtensor(gg.tensor("output_norm.weight")?, rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        if b != 1 {
            candle_core::bail!("qwen35 currently supports batch size 1, got {b}")
        }

        let mut last_hidden = None;
        for i in 0..l {
            let tok = input.narrow(1, i, 1)?;
            let mut h = self.embed_tokens.forward(&tok)?;
            for layer in &mut self.layers {
                h = layer.forward(&h, offset + i)?;
            }
            last_hidden = Some(h);
        }

        let h = self
            .norm
            .forward(&last_hidden.expect("at least one token"))?;
        let _enter = self.span_output.enter();
        self.lm_head.forward(&h)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
