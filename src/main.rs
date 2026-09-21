use std::f64::consts::E;

use axum::{
    Router,
    body::Body,
    http::StatusCode,
    response::{IntoResponse, Json, Response, Result},
    routing::get,
};
use chrono::{DateTime, TimeDelta, Utc};
use serde_json::{Value, json};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
mod schema;
mod server;

mod synthetic_data_provider;

mod simulation;

use crate::schema::{Order, Ticker};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ConnectBackend=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    let app = Router::new().route("/order", get(all_orders).post(create_order));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8050")
        .await
        .unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

/*
 * Initiate simulation and conditions()
 *  profile modeling (updates every timestep)
 *
 *
 *
 *
 *
 */


async fn all_orders() -> Result<impl IntoResponse> {
    return Ok("hello");
}

async fn create_order(Json(payload): Json<Value>) -> Result<impl IntoResponse> {
    let order: Value = json!(payload);
    match serde_json::from_value::<Order>(order) {
        Ok(order) => Ok(Json(order)),
        Err(e) => {
            return Err(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(e.to_string()))
                .unwrap()
                .into());
        }
    }
}
