use std::sync::Arc;
use chrono::{DateTime, TimeDelta, Utc, NaiveDate, Date};
use tokio::sync::Mutex;
use crate::schema::{Contract, Ticker, Order, Index};

mod stock_db;
mod market;

mod gjr_garch;

mod edge;


#[derive(Clone)]
pub struct Time(DateTime<Utc>);

impl Time {
    fn tick(&mut self, time_delta: TimeDelta) {
        self.0 += time_delta;
    }

    pub fn read(&self) -> DateTime<Utc> {
        self.0.clone()
    }

}
pub struct Simulation {
    time: Arc<Mutex<Time>>,
    start_date: DateTime<Utc>,
    end_date: DateTime<Utc>,
    portfolio: Vec<Ticker>,
    time_resolution: TimeDelta,
    account: Account,
    order_requests: Option<Vec<Order>>
}

struct Account {
    funds: f64,
    positions: Option<Vec<Position>>
}

impl Account {
    fn new(account_size: f64) -> Self {
        Self {
            funds: account_size,
            positions: None,
        }
    }
}
// Generally the broker will also have the multipliers and currency in the position record, but for our cases
struct Position {
    instrument: Instrument,
    quantity: i8,
    average_open_px: f64,
    current_mark: f64
}

impl Position {
    fn update_value(&mut self,current: f64) {
        self.current_mark = current;
    }
}

struct Instrument {
    ticker: Ticker,
    expiration_date: NaiveDate,
    strike: f32,
    option: Contract
}
impl Instrument {
    fn to_string(&self) -> String {
        let date: String = self.expiration_date.format("%Y%m%d").to_string();
        let option_type: String = match self.option{
            Contract::Call => "C".to_string(),
            Contract::Put => "P".to_string(),
        };
        let _strike: String = (self.strike * 1000.0).to_string();
        let mut price = "00000000".chars().collect::<Vec<char>>();
        for (i,c) in _strike.chars().rev().enumerate() {
            price[i] = c;
        }
        format!("{t} {d}{o}{p}", t=self.ticker, d=date,o=option_type,p=price.iter().collect::<String>())
    }

}
 impl Simulation {
     async fn initiate(portfolio: Vec<Ticker>,
                       time: Arc<Mutex<Time>>,
                 start_date: DateTime<Utc>,
                 end_date: DateTime<Utc>,
                 time_resolution: TimeDelta,
                 funds: f64
                 ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
         
         
         stock_db::data_loader(
               &portfolio,
               start_date,
               end_date,
               "5min", // right now this is assumed
               None
         ).await?;
         
         stock_db::index_loader(
             &vec![Index::GSPC,Index::VIX],
             start_date,
             end_date,
             "1h",
         ).await?;
         
         Ok(Self {
             time,
             start_date,
             end_date,
             portfolio,
             time_resolution,
             account: Account::new(funds),
             order_requests: None,
         })

     }

     async fn next(&mut self) {
         /*
         In principle, the main things that have to happen is that the progression of the simulation and therefore it's
         consequent data structure are being actively updated
          */
         self.time.lock().await.tick(self.time_resolution);


     }

 }