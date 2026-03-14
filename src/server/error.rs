use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct OpenAIError {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl OpenAIError {
    pub fn new(message: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: None,
            },
        }
    }

    pub fn internal(message: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: "internal_error".to_string(),
                param: None,
                code: None,
            },
        }
    }

    pub fn invalid_model(message: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: "invalid_request_error".to_string(),
                param: Some("model".to_string()),
                code: None,
            },
        }
    }
}

impl IntoResponse for OpenAIError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

impl From<anyhow::Error> for OpenAIError {
    fn from(err: anyhow::Error) -> Self {
        OpenAIError::internal(&err.to_string())
    }
}

impl From<std::io::Error> for OpenAIError {
    fn from(err: std::io::Error) -> Self {
        OpenAIError::internal(&err.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for OpenAIError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        OpenAIError::internal(&err.to_string())
    }
}
