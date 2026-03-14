use std::sync::Arc;

use axum::{extract::State, Json};

use crate::server::state::AppState;
use crate::server::types::{Model, ModelList};

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let model_ids = state.list_models().await;

    let models: Vec<Model> = if model_ids.is_empty() {
        vec![Model::new("default")]
    } else {
        model_ids.into_iter().map(|id| Model::new(&id)).collect()
    };

    Json(ModelList {
        object: "list".to_string(),
        data: models,
    })
}
