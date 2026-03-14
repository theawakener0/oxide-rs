use std::sync::Arc;

use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_service::Service;

use crate::server::router::create_router;
use crate::server::state::AppState;

pub async fn run(host: String, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).await?;

    println!();
    println!("┌─────────────────────────────────────────────┐");
    println!("│  oxide-rs - OpenAI Compatible Server        │");
    println!("│  Version: {}                        │", env!("CARGO_PKG_VERSION"));
    println!("└─────────────────────────────────────────────┘");
    println!();
    tracing::info!("Server starting on http://{}", addr);
    tracing::info!("API endpoints:");
    tracing::info!("  - POST /v1/chat/completions");
    tracing::info!("  - GET  /v1/models");
    tracing::info!("CORS: enabled (permissive)");
    println!();

    let state = Arc::new(AppState::new());
    let router = create_router(state);

    let cors = CorsLayer::permissive();
    let router = router.layer(cors);

    loop {
        let (stream, remote) = listener.accept().await?;
        tracing::debug!("Connection accepted from: {}", remote);

        let router = router.clone();

        tokio::spawn(async move {
            let io = TokioIo::new(stream);

            let service = hyper::service::service_fn(move |req| {
                let router = router.clone();
                async move { router.clone().call(req).await }
            });

            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::warn!("Connection error from {}: {}", remote, err);
            }
        });
    }
}

