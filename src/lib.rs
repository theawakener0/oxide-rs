pub mod cli;
pub mod inference;
pub mod model;

use std::path::Path;
use std::path::PathBuf;

pub use inference::{Generator, StreamEvent};
pub use model::{GgufMetadata, Model as ModelWrapper, TokenizerWrapper};

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub system_prompt: Option<String>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.3,
            top_p: None,
            top_k: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 299792458,
            system_prompt: None,
        }
    }
}

pub struct Model {
    generator: Option<Generator>,
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    options: GenerateOptions,
}

impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            generator: None,
            model_path: model_path.as_ref().to_path_buf(),
            tokenizer_path: None,
            options: GenerateOptions::default(),
        })
    }

    pub fn with_options(mut self, options: GenerateOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_tokenizer<P: AsRef<Path>>(mut self, tokenizer_path: P) -> Self {
        self.tokenizer_path = Some(tokenizer_path.as_ref().to_path_buf());
        self
    }

    pub fn load(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let generator = Generator::new(
            &self.model_path,
            self.tokenizer_path.as_ref(),
            self.options.temperature,
            self.options.top_p,
            self.options.top_k,
            self.options.seed,
            self.options.system_prompt.clone(),
        )?;
        self.generator = Some(generator);
        Ok(())
    }

    pub fn generate(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let generator = self
            .generator
            .as_mut()
            .ok_or("Model not loaded. Call load() first.")?;

        let result = generator.generate(
            prompt,
            self.options.max_tokens,
            self.options.repeat_penalty,
            self.options.repeat_last_n,
            |_event| {},
        )?;

        Ok(result)
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        mut callback: F,
    ) -> Result<String, Box<dyn std::error::Error>>
    where
        F: FnMut(String),
    {
        let generator = self
            .generator
            .as_mut()
            .ok_or("Model not loaded. Call load() first.")?;

        let mut output = String::new();
        generator.generate(
            prompt,
            self.options.max_tokens,
            self.options.repeat_penalty,
            self.options.repeat_last_n,
            |event| match event {
                StreamEvent::Token(t) => {
                    output.push_str(&t);
                    callback(t);
                }
                StreamEvent::Done => {}
            },
        )?;

        Ok(output)
    }

    pub fn warmup(&mut self, num_tokens: usize) -> Result<(), Box<dyn std::error::Error>> {
        let generator = self
            .generator
            .as_mut()
            .ok_or("Model not loaded. Call load() first.")?;
        generator.warmup(num_tokens)?;
        Ok(())
    }

    pub fn clear_history(&mut self) {
        if let Some(ref mut generator) = self.generator {
            generator.clear_history();
        }
    }

    pub fn metadata(&self) -> Option<&GgufMetadata> {
        self.generator.as_ref().map(|g| g.metadata())
    }
}

pub fn generate<P: AsRef<Path>>(
    model_path: P,
    options: GenerateOptions,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut model = Model::new(model_path)?.with_options(options);
    model.load()?;
    model.generate(prompt)
}
