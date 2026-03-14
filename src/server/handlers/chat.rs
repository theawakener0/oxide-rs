use std::sync::Arc;

use axum::{
    extract::State,
    response::{sse::Event, sse::Sse, IntoResponse, Response},
    Json,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::inference::StreamEvent;
use crate::server::error::OpenAIError;
use crate::server::state::AppState;
use crate::server::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChunkChoice, Choice,
    Delta, Usage, create_completion_id, get_timestamp,
};

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl axum::response::IntoResponse, OpenAIError> {
    let is_streaming = req.stream.unwrap_or(false);
    let request_id = create_completion_id();
    let model_path = req.model.clone();
    let temperature = req.temperature;
    let max_tokens = req.max_tokens;

    tracing::info!(
        "[{}] Request: POST /v1/chat/completions | model: {} | temp: {:.2} | max_tokens: {} | stream: {}",
        &request_id[..8],
        model_path,
        temperature,
        max_tokens,
        is_streaming
    );

    if is_streaming {
        Ok(ChatResponse::Streaming(handle_streaming(state, req, request_id).await?))
    } else {
        Ok(ChatResponse::NonStreaming(handle_non_streaming(state, req, request_id).await?))
    }
}

enum ChatResponse {
    NonStreaming(Json<ChatCompletionResponse>),
    Streaming(Sse<ReceiverStream<Result<Event, std::convert::Infallible>>>),
}

impl IntoResponse for ChatResponse {
    fn into_response(self) -> Response {
        match self {
            ChatResponse::NonStreaming(json) => json.into_response(),
            ChatResponse::Streaming(sse) => sse.into_response(),
        }
    }
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
) -> Result<Json<ChatCompletionResponse>, OpenAIError> {
    let model_path = &req.model;
    let generator = state.get_or_load_model(model_path).await?;

    let prompt = build_prompt(&req.messages);
    let completion_id = create_completion_id();
    let timestamp = get_timestamp();

    let repeat_penalty = 1.1f32;
    let repeat_last_n = 64usize;

    let prompt_tokens = prompt.split_whitespace().count();
    let mut completion_tokens = 0;
    let mut generated_text = String::new();

    tracing::info!(
        "[{}] Generation started | prompt: {} tokens | max: {}",
        &request_id[..8],
        prompt_tokens,
        req.max_tokens
    );

    let start_time = std::time::Instant::now();

    {
        let mut gen = generator.lock().map_err(|e| OpenAIError::internal(&e.to_string()))?;
        gen.generate_streaming(
            &prompt,
            req.max_tokens,
            repeat_penalty,
            repeat_last_n,
            |event| match event {
                StreamEvent::Token(token) => {
                    generated_text.push_str(&token);
                    completion_tokens += 1;
                }
                StreamEvent::PrefillStatus(_) => {}
                StreamEvent::Done => {}
            },
        )
        .map_err(|e| OpenAIError::internal(&e.to_string()))?;
    }

    let elapsed = start_time.elapsed();
    let tokens_per_sec = if elapsed.as_secs_f32() > 0.0 {
        completion_tokens as f32 / elapsed.as_secs_f32()
    } else {
        0.0
    };

    tracing::info!(
        "[{}] Generation complete | output: {} tokens | time: {:.2}s | speed: {:.1} tok/s",
        &request_id[..8],
        completion_tokens,
        elapsed.as_secs_f32(),
        tokens_per_sec
    );

    let response = ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion".to_string(),
        created: timestamp,
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: crate::server::types::ChatCompletionMessage {
                role: "assistant".to_string(),
                content: generated_text,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage::new(prompt_tokens, completion_tokens),
    };

    Ok(Json(response))
}

async fn handle_streaming(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
) -> Result<Sse<ReceiverStream<Result<Event, std::convert::Infallible>>>, OpenAIError> {
    let model_path = req.model.clone();
    let generator = state.get_or_load_model(&model_path).await?;

    let prompt = build_prompt(&req.messages);
    let completion_id = create_completion_id();
    let timestamp = get_timestamp();
    let repeat_penalty = 1.1f32;
    let repeat_last_n = 64usize;
    let max_tokens = req.max_tokens;

    let prompt_tokens = prompt.split_whitespace().count();

    tracing::info!(
        "[{}] Streaming started | prompt: {} tokens | max: {}",
        &request_id[..8],
        prompt_tokens,
        max_tokens
    );

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(100);

    let model_clone = req.model.clone();
    let request_id_clone = request_id.clone();

    std::thread::spawn(move || {
        let mut first = true;
        let mut completion_tokens = 0;
        let mut generated_text = String::new();
        let start_time = std::time::Instant::now();
        let mut first_token_time = None;
        
        let generator_result = generator.lock();
        if let Ok(mut gen) = generator_result {
            let result = gen.generate_streaming(
                &prompt,
                max_tokens,
                repeat_penalty,
                repeat_last_n,
                |event| match event {
                    StreamEvent::Token(token) => {
                        if first_token_time.is_none() {
                            first_token_time = Some(start_time.elapsed());
                        }
                        
                        generated_text.push_str(&token);
                        completion_tokens += 1;
                        
                        let delta = if first {
                            first = false;
                            Delta::with_role("assistant", &token)
                        } else {
                            Delta::new_content(&token)
                        };

                        let chunk = ChatCompletionChunk {
                            id: completion_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: timestamp,
                            model: model_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta,
                                finish_reason: None,
                            }],
                        };

                        let _ = tx.blocking_send(Ok(Event::default().json_data(chunk).unwrap()));
                    }
                    StreamEvent::PrefillStatus(_) => {}
                    StreamEvent::Done => {
                        let chunk = ChatCompletionChunk {
                            id: completion_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: timestamp,
                            model: model_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta::default(),
                                finish_reason: Some("stop".to_string()),
                            }],
                        };
                        let _ = tx.blocking_send(Ok(Event::default().json_data(chunk).unwrap()));

                        // Send complete message before [DONE]
                        let complete_response = ChatCompletionResponse {
                            id: completion_id.clone(),
                            object: "chat.completion".to_string(),
                            created: timestamp,
                            model: model_clone.clone(),
                            choices: vec![Choice {
                                index: 0,
                                message: crate::server::types::ChatCompletionMessage {
                                    role: "assistant".to_string(),
                                    content: generated_text.clone(),
                                },
                                finish_reason: Some("stop".to_string()),
                            }],
                            usage: Usage::new(prompt_tokens, completion_tokens),
                        };
                        let _ = tx.blocking_send(Ok(Event::default().json_data(complete_response).unwrap()));

                        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));

                        let elapsed = start_time.elapsed();
                        let tokens_per_sec = if elapsed.as_secs_f32() > 0.0 {
                            completion_tokens as f32 / elapsed.as_secs_f32()
                        } else {
                            0.0
                        };
                        let ttft = first_token_time.map(|t| t.as_secs_f32()).unwrap_or(0.0);

                        tracing::info!(
                            "[{}] Streaming complete | output: {} tokens | time: {:.2}s | ttft: {:.2}s | speed: {:.1} tok/s",
                            &request_id_clone[..8],
                            completion_tokens,
                            elapsed.as_secs_f32(),
                            ttft,
                            tokens_per_sec
                        );
                    }
                },
            );

            if let Err(e) = result {
                let _ = tx.blocking_send(Ok(Event::default().data(format!("Error: {}", e))));
            }
        } else {
            let _ = tx.blocking_send(Ok(Event::default().data("Error: Failed to acquire generator lock")));
        }
    });

    let stream = ReceiverStream::new(rx);

    Ok(Sse::new(stream))
}

fn build_prompt(messages: &[crate::server::types::ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("system: ");
            }
            "user" => {
                prompt.push_str("user: ");
            }
            "assistant" => {
                prompt.push_str("assistant: ");
            }
            _ => {
                prompt.push_str(&format!("{}: ", msg.role.as_str()));
            }
        }
        prompt.push_str(&msg.content);
        prompt.push_str("\n");
    }

    prompt.push_str("assistant: ");
    prompt
}
