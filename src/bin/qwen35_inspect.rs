use std::fs::File;
use std::io::Cursor;

use candle_core::quantized::gguf_file;
use memmap2::Mmap;

fn main() -> anyhow::Result<()> {
    let path = "/home/theawakener/Projects/OpenEye_00/models/Qwen3.5-0.8B-Q4_K_M.gguf";
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap);
    let content = gguf_file::Content::read(&mut cursor)?;

    for key in [
        "general.architecture",
        "qwen35.attention.key_length",
        "qwen35.rope.freq_base",
        "qwen35.rope.dimension_sections",
        "qwen35.attention.layer_norm_rms_epsilon",
        "qwen35.ssm.inner_size",
        "qwen35.ssm.group_count",
    ] {
        println!("{key}: {:?}", content.metadata.get(key));
    }

    for key in [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_norm.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
    ] {
        println!(
            "{key}: {:?}",
            content.tensor_infos.get(key).map(|t| t.shape.dims())
        );
    }

    Ok(())
}
