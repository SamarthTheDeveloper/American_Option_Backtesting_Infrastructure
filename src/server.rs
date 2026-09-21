use std::io::{self, Write};

use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming, transport::Server};

use quote_service::quote_server::{Quote, QuoteServer};
use quote_service::{ChainRequest, QuoteRequest, QuoteResponse};

use market_service::{ProgRequest, ProgStatus};

pub mod quote_service {
    tonic::include_proto!("quote");
}

pub mod market_service {
    tonic::include_proto!("market");
}


#[derive(Debug, Default)]
struct QuoteProvider {}

type Stream = ReceiverStream<Result<QuoteResponse, Status>>;

#[tonic::async_trait]
impl Quote for QuoteProvider {
    type StreamChainStream = Stream; // TODO: change the name
    /*
    In gRPC the client runs the server functions as if they were local.

     */
    async fn get_quote(&self, request: Request<QuoteRequest>) -> Result<Response<QuoteResponse>, Status>{
        let parameters = request.into_inner();



        let resp = QuoteResponse {
            contract: format!("call on {}", parameters.symbol).to_string(),
            strike: 0.0,
            bid: 0.0,
            ask: 0.0,
            iv: 0.0,
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0
        };
        Ok(Response::new(resp))
    }

    async fn stream_chain(&self,
                          request: Request<Streaming<ChainRequest>>)
        -> Result<Response<Self::StreamChainStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            while let Some(msg) = inbound.next().await {
                let request = match msg {
                    Ok(msg) => msg,
                    Err(e) => {
                        println!("Inbound error: {:?}", e);
                        break;
                    }
                };
                let message = QuoteResponse {
                    contract: "call".to_string(),
                    strike: 0.0,
                    bid: 0.0,
                    ask: 0.0,
                    iv: 0.0,
                    delta: 0.0,
                    gamma: 0.0,
                    theta: 0.0,
                    vega: 0.0,
                    rho: 0.0
                };
                if tx.send(Ok(message)).await.is_err() {
                    println!("Outbound error");
                }
            }

        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }

}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /*
    TODO: Setup actual gRPC server for streaming
     option chain quotes and market simulation to cloud gpu client
     */
    /*
        Take a Request: Either next time step or quote
        Send a data Response,
        Next Time Step:
        - Account value
        - Positions:
        - Orders pending:
        - Set of the past month of the portfolio stocks HLOC data

    */
    let addr = "[::]:50051".parse()?;
    let quote_provider = QuoteProvider::default();

    Server::builder()
        .add_service(QuoteServer::new(quote_provider))
        .serve(addr)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    pub mod test_service {
        tonic::include_proto!("test");
    }
    use test_service::delivery_server::{Delivery, DeliveryServer};
    use test_service::{TextRequest, TextResponse};
    // Basic gRPC streaming backend setup for testing; streams strings with the test protobuffer
    // Consider testing streaming video and image with chunking and byte serialization

    fn input(prompt: &str) -> String { // helper function
        print!("{}", prompt);
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        line.trim_end().to_string()
    }
    #[derive(Debug, Default)]
    struct Deliverer {}
    type Stream = ReceiverStream<Result<TextResponse, Status>>;
    #[tonic::async_trait]
    impl Delivery for Deliverer {
        type ChatStream = Stream;

        async fn send(&self, request: Request<TextRequest>) -> Result<Response<TextResponse>, Status> {
            println!("Message: {:?}", request);
            let reply = TextResponse {
                text: "If you're seeing this then we're good".to_string(),
            };
            Ok(Response::new(reply))
        }

        async fn chat(
            &self,
            request: Request<Streaming<TextRequest>>,
        ) -> Result<Response<Self::ChatStream>, Status> {
            let mut inbound = request.into_inner();
            let (tx, rx) = mpsc::channel(32);
            tokio::spawn(async move {
                while let Some(msg) = inbound.next().await {
                    let request = match msg {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("Inbound stream error: {}", e);
                            break;
                        }
                    };
                    if !request.text.is_empty() {
                        println!("Request: {}", request.text);
                    }
                    let message = TextResponse {
                        text: input("Response: "),
                    };
                    if tx.send(Ok(message)).await.is_err() {
                        println!("Failed to send response");
                        break;
                    }
                }
                println!("Live chat ended");
            });
            Ok(Response::new(ReceiverStream::new(rx)))
        }
    }

    #[tokio::test]
    async fn chat_system() -> Result<(),Box<dyn std::error::Error>> {
        let addr = "[::]:50051".parse()?;
        let deliverer = Deliverer::default();

        Server::builder()
            .add_service(DeliveryServer::new(deliverer))
            .serve(addr)
            .await?;
        Ok(())

    }
}
