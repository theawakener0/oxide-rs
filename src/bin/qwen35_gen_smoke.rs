use std::path::PathBuf;

use oxide_rs::inference::{Generator, StreamEvent};

fn main() -> anyhow::Result<()> {
    let model_path =
        PathBuf::from("/home/theawakener/Projects/OpenEye_00/models/Qwen3.5-0.8B-Q4_K_M.gguf");
    let mut generator = Generator::new(&model_path, None, 0.0, None, None, 299792458, None, 128)?;
    generator.generate_streaming("How are you?", 32, 1.0, 64, |event| match event {
        StreamEvent::Token(t) => print!("{}", t),
        StreamEvent::Done => println!(),
        StreamEvent::PrefillStatus(_) => {}
    })?;
    Ok(())
}
