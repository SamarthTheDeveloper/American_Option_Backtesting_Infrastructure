use crate::schema::{Index, Profile, Ticker};
use crate::simulation::stock_db::{fetch_stock_data, fetch_index_data};
use chrono::{DateTime, TimeDelta, Utc, NaiveDate, Date};
use stochastic_rs::stats::realized::log_returns;
use crate::simulation::{edge, Time};
use crate::simulation::gjr_garch::GjrGarch;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use super::Simulation;
pub struct Market {
    time: Arc<super::Time>,
    sim: Arc<Simulation>,
    profiles: HashMap<Ticker, Profile>,
    trackers: Vec<Tracker>,
    status: Status
}

impl Market {
    
    pub fn start(&mut self) {
        self.profiles = HashMap::new();
        self.trackers = Vec::new();
        self.status = Status::SETUP;
    }

    fn set_trackers(&mut self, trackers: Vec<Tracker>) {
        self.trackers = trackers;
    }
    fn set_impressions(&mut self, ticker: Ticker, p: Profile) {
        self.profiles.insert(ticker, p);
    }
}
pub enum Status {
    ACTIVE,
    IDLE,
    SETUP
}

enum Tracker {
    VIABLE{
        time: Arc<Time>,
        start: DateTime<Utc>,
        cycle: TimeDelta
    },
    UPDATE{
        time: Arc<Time>,
        cycle: TimeDelta
    }
}

impl Tracker {

    pub fn new( time: Arc<Time>,start: DateTime<Utc>, cycle: TimeDelta ) -> Self {
        Self::VIABLE {
            time,
            start,
            cycle
        }
    }

    pub fn track(&mut self) {
        if let Self::VIABLE { time, start, cycle } = self {
            if time.read() - *start == *cycle {
                let time = time.clone();
                let cycle = cycle.clone();
                *self = Self::UPDATE { time,cycle  };
            }
            return;
        }

        if let Self::UPDATE { time,cycle } = self {
            let time = time.clone();
            let start = time.read();
            let cycle = cycle.clone();
            *self = Self::VIABLE { time, start, cycle };
        }
    }

}
impl Market {

    fn log_returns(o: &Vec<f64>) -> Vec<f64> {
        let mut returns = vec![0.0; o.len()];
        for (i, x ) in o.into_iter().skip(1).enumerate() {
             returns[i] = (x/o[i-1]).ln();
        }
        returns
    }

    fn simulate(g: &GjrGarch, eps: f64) -> f64{
        let mut s2 = g.seed_var();
        s2 = g.next_var(eps, s2);
        eps + g.mu
    }

    fn volatility_risk_prem(date: DateTime<Utc>) -> f64 {

        let mut spx = fetch_index_data(&Index::GSPC,date- TimeDelta::days(31), Some(date)).unwrap();
        let vix = fetch_index_data(&Index::VIX,date, None).unwrap();
        let adj: Vec<f64> = spx.iter_mut().map(|l| l.adj_close ).collect::<Vec<f64>>();
        let sr: f64 = Self::log_returns(&adj)
            .into_iter()
            .map(|x| x*x)
            .sum();
        let rv: f64 = 100.0 * ((252.0/30.0) * sr).sqrt();

        vix[0].adj_close / rv
    }

    pub(super) fn run(&mut self,sim: &Simulation) {
        
        // TODO: Use Rayon parallel iteration par_iter()
        match self.status {
            Status::SETUP => {
                sim.portfolio.iter().for_each(| ticker | {
                    let data = fetch_stock_data(ticker,
                                                sim.start_date - TimeDelta::days(180),
                                                Some(sim.start_date));
                    let (mut o, mut h, mut l, mut c) = (vec![], vec![], vec![], vec![]);

                    let mut n = 1;
                    let mut m = 0.0;

                    data.unwrap().iter().for_each(|stock| {
                        h.push(stock.high);
                        o.push(stock.open);
                        l.push(stock.low);
                        c.push(stock.close);
                        m = ((n as f64) * m + stock.open) / (n + 1) as f64;
                        n += 1;
                    });

                    let _ = n;

                    let bsp = edge::edge(&o, &h, &l, &c, false).unwrap();

                    let log_r = Self::log_returns(&o);

                    let garch: GjrGarch = GjrGarch::fit(&log_r);


                    self.set_impressions(ticker.clone(), Profile {
                        omega: garch.omega,
                        alpha: garch.alpha,
                        beta: garch.beta,
                        bsp,
                        last_var: Self::simulate(&garch, 0.0),
                        prev_eps: log_r[log_r.len() - 1] - m,
                        volatility_risk_premium: Self::volatility_risk_prem(sim.start_date), // get Vix
                        unconditional_volatility: garch.uncond_var().unwrap(),
                        periods_per_year: 19656
                    });

                    self.set_trackers(vec![
                        Tracker::new(self.time.clone(), sim.start_date, TimeDelta::days(30)),
                        Tracker::new(self.time.clone(), sim.start_date, TimeDelta::days(1)),
                        Tracker::new(self.time.clone(), sim.start_date, TimeDelta::days(7))
                    ]);

                })

            }
            Status::ACTIVE => {
                
            }
            Status::IDLE => {}
        }
    }
}