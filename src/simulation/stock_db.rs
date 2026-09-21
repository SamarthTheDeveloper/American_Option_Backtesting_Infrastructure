use bytes::Bytes;
use chrono::{DateTime, TimeDelta, Utc};
use dotenv::dotenv;
use hyper::{Request, Response, http::Method};
use hyper_util::rt::TokioIo;
use http_body_util::{Empty, BodyExt};
use hyper::client::conn::http1::SendRequest;
use rusqlite::{Connection, params};
use std::sync::Arc;
use tokio_rustls::TlsConnector;
use tokio_rustls::rustls::{ClientConfig, RootCertStore, pki_types::ServerName};
use crate::schema::{Stock, Ticker, Index, Level};
use serde_json::from_slice;
use serde::Deserialize;
use std::env;
use std::time::Duration;
use tokio::net::TcpStream;

pub(super) fn fetch_stock_data(
    ticker: &Ticker,
    start_date: DateTime<Utc>,
    end_date: Option<DateTime<Utc>>,
) -> Result<Vec<Stock>, rusqlite::Error> {
    let conn = Connection::open("app.db")?;
    match end_date {
        Some(end_date) => {
            let mut stmt = conn
                .prepare("SELECT * FROM ?1 where date BETWEEN ?2 AND ?3")?;
            let rows: Vec<Stock> = stmt
                .query_map(
                    params![
                        format!("{ticker}"),
                        start_date.to_rfc3339(),
                        end_date.to_rfc3339()
                    ],
                    Stock::from_row,
                )?
                .collect::<Result<Vec<Stock>, rusqlite::Error>>()?;
            Ok(rows)
        }
        None => {
            let price = conn.query_row(
                "SELECT * FROM ?1 WHERE date = ?2",
                params![format!("{ticker}"), start_date.to_rfc3339()],
                Stock::from_row,
            )?;
            Ok(vec![price])
        }
    }
}

pub(super) fn fetch_index_data(
    index: &Index,
    start_date: DateTime<Utc>,
    end_date: Option<DateTime<Utc>>,
) -> Result<Vec<Level>, rusqlite::Error> {
    let conn = Connection::open("app.db")?;
    match end_date {
        Some(end_date) => {
            let mut stmt = conn
                .prepare("SELECT * FROM ?1 where date BETWEEN ?2 AND ?3")?;
            let rows: Vec<Level> = stmt
                .query_map(
                    params![
                        format!("{index}"),
                        start_date.to_rfc3339(),
                        end_date.to_rfc3339()
                    ],
                    Level::from_row,
                )?
                .collect::<Result<Vec<Level>, rusqlite::Error>>()?;
            Ok(rows)
        }
        None => {
            let price = conn.query_row(
                "SELECT * FROM ?1 WHERE date = ?2",
                params![format!("{index}"), start_date.to_rfc3339()],
                Level::from_row,
            )?;
            Ok(vec![price])
        }
    }
}

type Sender = SendRequest<Empty<Bytes>>;
type BoxError = Box<dyn std::error::Error + Send + Sync>;

async fn connect(host: &str) -> Result<Sender, BoxError> {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let connector = TlsConnector::from(Arc::new(config));

    let stream = TcpStream::connect((host, 443)).await?;
    let domain = ServerName::try_from(host)?.to_owned();
    let tls_stream = connector.connect(domain, stream).await?;

    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(tls_stream)).await?;
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("TLS connection error: {}", e);
        }
    });
    Ok(sender)
}
pub(super) async fn data_loader(
    stocks: &Vec<Ticker>,
    start_date: DateTime<Utc>,
    end_date: DateTime<Utc>,
    resample_freq: &str,
    trailing_date: Option<TimeDelta>,
) -> Result<(), BoxError> {
    let trailing_date = trailing_date.unwrap_or(TimeDelta::days(180));
    let conn = Connection::open("app.db").unwrap();
    let start_date = start_date - trailing_date;
    let token = env::var("TIINGO_API_KEY").expect("tiingo api key not set");
    let mut sender = connect("api.tiingo.com").await?;
    for stock in stocks {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ?1 (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
          )",
            params![format!("{stock}")],
        )?;
        let mut stock_data = stock_api(&mut sender, &token, &stock, start_date, end_date, resample_freq).await?;
        for data in stock_data {
            conn.execute(
                "INSERT OR REPLACE INTO ?1 (date, open, high, low, close, volume) VALUES (?2, ?3, ?4, ?5, ?6, ?7)",
                params![format!("{stock}"), data.date.to_rfc3339(), data.open, data.high, data.low, data.close, data.volume],
            )?;
        }
    }
    Ok(())
}

pub(super) async fn index_loader(
    indexes: &Vec<Index>,
    start_date: DateTime<Utc>,
    end_date: DateTime<Utc>,
    resample_freq: &str,
) -> Result<(), BoxError> {

    let conn = Connection::open("app.db").unwrap();
    let mut sender = connect("query1.finance.yahoo.com").await?;

    for index in indexes {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ?1 (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adjclose REAL
            )",
            params![format!("{index}")],

        )?;
        let mut index_data = index_api(&mut sender, &index, start_date, end_date, resample_freq).await?;
        for data in index_data {
            conn.execute(
                "INSERT OR REPLACE INTO ?1 (date, open, high, low, close, adjclose) VALUES (?2, ?3, ?4, ?5, ?6, ?7)",
                params![format!("{index}"), data.period.to_rfc3339(), data.open, data.high, data.low, data.close, data.adj_close]
            )?;
        }
    }
    Ok(())

}

async fn stock_api(
    sender: &mut Sender,
    token: &str,
    stock: &Ticker,
    start_date: DateTime<Utc>,
    end_date: DateTime<Utc>,
    resample_freq: &str,
) -> Result<Vec<Stock>, BoxError> {
    dotenv().ok();

    let base_url = format!(
        "/iex/{t}/prices?startDate={sd}&endDate={ed}&resampleFreq={r}&columns=open,high,low,close,volume",
        t = stock,
        sd = start_date.date_naive(),
        ed = end_date.date_naive(),
        r = resample_freq
    );

    let req = Request::builder()
        .method(Method::GET)
        .uri(base_url)
        .header("Host", "api.tiingo.com")
        .header("Authorization", format!("Token {}", token))
        .header("Accept", "application/json")
        .body(Empty::<Bytes>::new())?;

    sender.ready().await?;
    let resp = sender.send_request(req).await?;
    let bytes = resp.into_body().collect().await?.to_bytes();
    let data: Vec<Stock> = serde_json::from_slice(&bytes)?;
    Ok(data)
}

async fn index_api(
    sender: &mut Sender,
    index: &Index,
    start_date: DateTime<Utc>,
    end_date: DateTime<Utc>,
    interval: &str
) -> Result<Vec<Level>, BoxError> {

    let base_url = format!(
        "/v8/finance/chart/{i}?period1={p1}&period2={p2}&interval={f}",
        i = index,
        p1 = start_date.timestamp(),
        p2 = end_date.timestamp(),
        f = interval
    );

    let req = Request::builder()
        .method(Method::GET)
        .uri(base_url)
        .header("Host", "query1.finance.yahoo.com")
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
                       AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")
        .header("Accept", "application/json")
        .body(Empty::<Bytes>::new())?;

    sender.ready().await?;
    let resp = sender.send_request(req).await?;
    let status = resp.status();
    let bytes = resp.into_body().collect().await?.to_bytes();
    //eprintln!("status={status} len={} body={:?}", bytes.len(), String::from_utf8_lossy(&bytes));
    let data: serde_json::Value = serde_json::from_slice(&bytes)?; // this won't work but this is the placeholder

    // fix
    let period = data.pointer("/chart/result/0/timestamp").unwrap();
    let nadjclose = data.pointer("/chart/result/0/indicators/adjclose/0/adjclose").unwrap();
    let nclose = data.pointer("/chart/result/0/indicators/quote/0/close").unwrap();
    let nopen = data.pointer("/chart/result/0/indicators/quote/0/open").unwrap();
    let nhigh = data.pointer("/chart/result/0/indicators/quote/0/high").unwrap();
    let nlow = data.pointer("/chart/result/0/indicators/quote/0/low").unwrap();

    let period: Vec<i64> = Vec::deserialize(period)?;
    let adjclose: Vec<f64> = Vec::deserialize(nadjclose)?;
    let close: Vec<f64> = Vec::deserialize(nclose)?;
    let open: Vec<f64> = Vec::deserialize(nopen)?;
    let high: Vec<f64> = Vec::deserialize(nhigh)?;
    let low: Vec<f64> = Vec::deserialize(nlow)?;

    let mut levels: Vec<Level> = Vec::with_capacity(period.len());

    for i in 0..period.len() {
       levels.push(
           Level{
               period: DateTime::from_timestamp(period[i],0).expect("invalid period"),
               open: open[i],
               close: close[i],
               high: high[i],
               low: low[i],
               adj_close: adjclose[i]
           }
       )
    }

    Ok(levels)

}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn yfinance_index_api() {

        let mut sender = connect("query1.finance.yahoo.com").await.unwrap();

        let p1: DateTime<Utc> = DateTime::parse_from_rfc3339("2026-08-03T13:30:00.000Z")
            .expect("Failed to parse")
            .to_utc();
        dbg!(p1.timestamp());
        let p2: DateTime<Utc> = DateTime::parse_from_rfc3339("2026-08-31T20:00:00.000Z")
            .expect("Failed to parse")
            .to_utc();
        dbg!(p2.timestamp());

        let spx = index_api(&mut sender, &Index::GSPC, p1, p2, "1d").await;

        match spx {
            Ok(data) => {
                println!("{:#?}", data);
            },
            Err(err) => {
                eprintln!("Oh No!: {:?}", err);
            }
        }
    }

    #[tokio::test]
    async fn tiingo_stock_api() {
        match dotenv().ok() {
            Some(_) => {
                println!("env was found");
            }
            None => {
                println!("no env found");
            }
        }

        let token = env::var("TIINGO_API_KEY").expect("tiingo api key not set");
        let mut sender = connect("api.tiingo.com").await.unwrap();

        let p1: DateTime<Utc> = DateTime::parse_from_rfc3339("2026-09-08T14:00:00.000Z")
            .expect("Failed to parse")
            .to_utc();
        let p2: DateTime<Utc> = DateTime::parse_from_rfc3339("2026-09-11T19:00:00.000Z")
            .expect("Failed to parse")
            .to_utc();

        let nflx = stock_api(&mut sender, &token, &Ticker::NFLX, p1, p2, "1hour").await;

        match nflx {
            Ok(data) => { println!("{:#?}", data); }
            Err(err) => {
                eprintln!("Oh No!: {:?}", err);
            }
        }
    }
}
