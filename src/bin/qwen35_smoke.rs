use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path =
        PathBuf::from("/home/theawakener/Projects/OpenEye_00/models/Qwen3.5-0.8B-Q4_K_M.gguf");
    let mut model = oxide_rs::model::Model::load(&path)?;
    let logits = model.forward(&[1u32], 0)?;
    println!("loaded {} {:?}", model.metadata().name, logits.dims());
    Ok(())
}
