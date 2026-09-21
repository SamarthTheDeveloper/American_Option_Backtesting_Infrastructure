use chrono::{DateTime, Utc};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use strum_macros::Display;
use uuid::Uuid;
#[derive(Builder, Serialize, Deserialize, Clone)]
pub struct Order {
    pub order_id: Uuid,
    pub symbol: String,
    pub contract_type: Contract,
    #[serde(default)]
    pub legs: Option<Vec<Leg>>,
    pub quantity: u64,
    pub status: OrderStatus,
    #[serde(default)]
    pub stop_price: Option<u64>,
    #[serde(default)]
    pub limit_price: Option<u64>,
    pub order_class: OrderClass,
    pub order_type: OrderType,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub submitted_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub cancelled_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub expired_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub filled_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub filled_quantity: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Leg {
    #[serde(flatten)]
    pub order: Order,
    pub position_intent: PositionIntent,
    pub ratio_quantity: u64,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum PositionIntent {
    BuyToOpen,
    SellToOpen,
    BuyToClose,
    SellToClose,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum OrderStatus {
    Pending,
    Filled,
    Cancelled,
    Expired,
    Replaced,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum OrderType {
    Limit,
    Market,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum OrderClass {
    SIMPLE,
    MLEG,
    OCO, // OCO to Bracket are offered only for equity not options
    OTO,
    Bracket,
}

pub enum Position {
    Long,
    Short,
}
#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum Contract {
    Put,
    Call,
}

pub struct Profile {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub bsp: f64,
    pub last_var: f64,
    pub prev_eps: f64,
    pub volatility_risk_premium: f64,
    pub unconditional_volatility: f64,
    pub periods_per_year: u32, // should this be 252?
}

impl Profile {
    pub fn calibrate(&mut self) {}
}

pub struct Fundamental {
    date: DateTime<Utc>,
    delta: f64,
    bond_rate: f64,
    dividend_yield: f64,
    standard_deviation: f64,
    volume: f64,
    open: f64,
    close: f64,
}

pub struct Greeks {
    pub delta: f64,
    gamma: f64,
    pub vega: f64,
    theta: f64,
    rho: f64,
}

impl Greeks {
    pub fn new(delta: f64, gamma: f64, vega: f64, theta: f64, rho: f64) -> Self {
        Self {
            delta,
            gamma,
            vega,
            theta,
            rho,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Level {
    pub period: DateTime<Utc>,
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub adj_close: f64
}


#[derive(Serialize, Deserialize, Debug)]
pub struct Stock {
    pub date: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}



impl Stock {
    pub fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        let s: String = row.get(0)?;
        Ok(Self {
            date: DateTime::parse_from_rfc3339(&s)
                .unwrap()
                .with_timezone(&Utc),
            open: row.get(1)?,
            high: row.get(2)?,
            low: row.get(3)?,
            close: row.get(4)?,
            volume: row.get(5)?,
        })
    }
}

impl Level {
    pub fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        let s: String = row.get(0)?;
        Ok(Self {
            period: DateTime::parse_from_rfc3339(&s)
                .unwrap()
                .with_timezone(&Utc),
            open: row.get(1)?,
            high: row.get(2)?,
            low: row.get(3)?,
            close: row.get(4)?,
            adj_close: row.get(5)?,
        })
    }
}

pub struct Quote {
    contract: Contract,
    strike: f64,
    bid: f64,
    ask: f64,
    iv: f64,
    delta: f64,
    gamma: f64,
    theta: f64,
    vega: f64,
    rho: f64,
}

impl Quote {
    pub fn new(
        contract: Contract,
        strike: f64,
        bid: f64,
        ask: f64,
        iv: f64,
        greeks: Greeks,
    ) -> Self {
        Self {
            contract,
            strike,
            bid,
            ask,
            iv,
            delta: greeks.delta,
            gamma: greeks.gamma,
            theta: greeks.theta,
            vega: greeks.vega,
            rho: greeks.rho,
        }
    }
}


#[derive(Debug, Display)]
#[strum(serialize_all = "UPPERCASE", prefix = "%5E")]
pub enum Index {
    GSPC,
    VIX,
    IXIC,
    NDX,
    RUI,
    RUT,
    DJI
}
#[derive(Debug, Display, Clone, Copy, Eq, Hash, PartialEq)]
#[strum(serialize_all = "lowercase")]
pub enum Ticker {
    AFRM,
    HOOD,
    NFLX,
    DAL,
    UAL,
    CVS,
    GM,
    PDD,
    CCJ,
    XBI,
    XHB,
    EEM,
    KO,
    ZEIM,
    TEVA,
    SMCI,
    AA,
    BMY,
}

enum StrikeProgram {
    Standard,
    OneDollar,
    FiftyCent,
}

enum MPV {
    OneCent,
    FiveCent,
    TenCent,
}

impl MPV {
    fn increment(&self) -> f64 {
        match self {
            MPV::OneCent => 0.01,
            MPV::FiveCent => 0.05,
            MPV::TenCent => 0.1,
        }
    }
}

impl Ticker {
    fn strike_program(&self) -> StrikeProgram {
        match self {
            Ticker::NFLX => StrikeProgram::OneDollar,
            Ticker::HOOD => StrikeProgram::FiftyCent,
            _ => StrikeProgram::Standard,
        }
    }
    // fn liquidity_anchor(&self) -> f64 {
    //     match self {

    //     }
    // }
}

