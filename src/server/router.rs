use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};

use crate::server::handlers::{chat_completions, list_models};
use crate::server::state::AppState;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .with_state(state)
}
