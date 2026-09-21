use crate::schema::{Contract, Greeks, Profile, Quote, Ticker};
use chrono::{DateTime, TimeDelta, Utc};
use statrs::distribution::{ContinuousCDF, Normal};

/*
* Pricing skeleton

 -Volatility surface, not a constant vol — skew/smile across strikes + term structure across expiries(check)
 -Arbitrage-free parametrization (SVI or SABR) with calendar-arb and butterfly-arb constraints enforced(check)
 -Surface level tied to realized vol of the real path + variance risk premium (IV > RV by a few points)(check )
 -Leverage effect — IV spikes on big down moves in the underlying (Check)
 -Forward-based pricing: F = S·e^((r−q)T) with a real rate curve + real dividend schedule (Check)
 -American-style numerics for single names (binomial / Bjerksund-Stensland), early exercise around dividends (Check but discrete dividends put in hold)
 -Put-call parity enforced — one fair value per strike, derive both legs (don't generate calls/puts independently) (Check)

Microstructure

 -Bid-ask spread as a function of vega, price level, moneyness, DTE — never constant (ignore)
 -Percentage-spread blowup for cheap OTM options (a $0.05 option quoting $0.05 × $0.10)
 -Spread widening at both DTE extremes (0DTE and far-dated)
 -Tick rounding to valid increments ($0.01 < $3, $0.05 above; per penny-pilot class)
 -Quote staleness / update latency — deep-OTM and far-dated strikes don't refresh every underlying tick
 -Liquidity structure if synthesizing volume/OI — concentration at ATM and round strikes, near expiries; most strikes zero-volume; OI accreting over time (Poisson arrivals scaled by moneyness/DTE)

Randomness / noise

 -Per-strike IV jitter correlated across adjacent strikes (preserves convexity) — not i.i.d. per strike
 -Stochastic, regime-dependent spread width (spreads widen in stress)
 -Day-to-day surface dynamics — ATM vol as mean-reverting (OU/Heston-style) process, skew steepening/flattening, term-structure tilt
 -Vol innovations negatively correlated with underlying returns
 -Trade-side randomness if emitting prints — hit bid/ask stochastically per imbalance, occasionally mid or through
 -Earnings IV-crush cycle for single names — ramp into the event, collapse after
*
*
*/
pub fn option_chain_quote(
    // use the same IV in call as put for same strike
    ticker: Ticker,
    strike: f64,
    stock: f64,
    date: DateTime<Utc>,
    expiration: DateTime<Utc>,
    dividend_yield: f64,
    interest_rate: f64,
    profile: Profile,
) -> Result<(Quote, Quote), Box<dyn std::error::Error>> {
    let S = stock;
    let r = interest_rate;
    let q = dividend_yield;
    let ppy = profile.periods_per_year;
    let λ = 3.0;
    let σ_svi: f64 = 0.10;
    let m = 0.05;
    let T_period = (expiration - date).num_days() as f64 / 365.0;
    let omega = profile.omega;
    let alpha = profile.alpha;
    let beta = profile.beta;
    let variance = profile.last_var;
    let VRP = profile.volatility_risk_premium;
    let uncond_vol = profile.unconditional_volatility; // sigma bar
    let garch = (variance * ppy as f64).sqrt(); // sigma garch
    let σ_T: f64 = uncond_vol + (garch * VRP - uncond_vol) * (-λ * T_period).exp();
    let a = σ_T.powi(2);
    let b = 0.15 * a * (0.20 / σ_T);
    let p = (-0.60 - (σ_T - 0.20)).max(-0.95);
    let w = |k: f64| a + b * (p * (k - m) + ((k - m).powi(2) + σ_svi.powi(2)).sqrt()); // Total Implied Variance
    let F = S * (T_period * (r - q)).exp();
    let k = (strike / F).ln();
    let iv = (w(k) / T_period).sqrt();
    /*
    *
    * option: Contract,
    strike: f64,
    date: DateTime<Utc>,
    expiration: DateTime<Utc>,
    sigma: f64,
    r: f64,
    fundamental: Fundamental ,
    */
    let cp_mid = bjerksund_stensland_quote(Contract::Call, strike, date, expiration, k, r, q, S);
    let pp_mid = bjerksund_stensland_quote(Contract::Put, strike, date, expiration, k, r, q, S);
    // k = ln(K/F)
    //
    let cgreeks = option_greeks(Contract::Call, strike, S, date, expiration, k, r, q);
    let pgreeks = option_greeks(Contract::Put, strike, S, date, expiration, k, r, q);

    let calculate_spread = |P_mid: f64, greeks: &Greeks| {
        let SPRund = S * profile.bsp * (garch / uncond_vol);
        let cost_delta = greeks.delta.abs() * SPRund;
        let cost_vega = greeks.vega * 0.05;
        let cost_cap = P_mid * 0.005;
        let spread_total = cost_delta + cost_vega + cost_cap;
        let tick_min = if S < 3.00 { 0.01 } else { 0.05 };
        let spread_final = spread_total.max(tick_min);
        let bid = P_mid - spread_final / 2.0;
        let ask = P_mid + spread_final / 2.0;
        (bid, ask)
    };

    let (cbid, cask) = calculate_spread(cp_mid, &cgreeks);
    let (pbid, pask) = calculate_spread(pp_mid, &pgreeks);

    Ok((
        Quote::new(Contract::Call, strike, cbid, cask, iv, cgreeks),
        Quote::new(Contract::Put, strike, pbid, pask, iv, pgreeks),
    ))
    // Microstructuring

    /* Microstructuring Chechlist
     * Rolls Model
     * GJR-Garch(1,1)
     * m
     */
}

fn option_greeks(
    o: Contract,
    k: f64,
    s: f64,
    d: DateTime<Utc>,
    e: DateTime<Utc>,
    v: f64,
    r: f64,
    q: f64,
) -> Greeks {
    let h = s * 10.0_f64.powf(-4.0);
    let h_v = 0.01;
    let h_t = TimeDelta::seconds(1);
    let h_t_f64 = 1.0 / (365.0 * 24.0 * 3600.0);
    let h_r = 0.0001;
    let bs = |s: f64, d: DateTime<Utc>, v: f64, r: f64| {
        bjerksund_stensland_quote(o.clone(), k, d, e, v, r, q, s)
    };
    let delta = (bs(s + h, d, v, r) - bs(s - h, d, v, r)) / (2.0 * h);
    let gamma = (bs(s + h, d, v, r) - 2.0 * bs(s, d, v, r) + bs(s - h, d, v, r)) / (h.powi(2));
    let theta = (bs(s, d + h_t, v, r) - bs(s, d - h_t, v, r)) / (2.0 * h_t_f64);
    let vega = (bs(s, d, v + h_v, r) - bs(s, d, v - h_v, r)) / (2.0 * h_v);
    let rho = (bs(s, d, v, r + h_r) - bs(s, d, v, r - h_r)) / (2.0 * h_r);
    Greeks::new(delta, gamma, theta, vega, rho)
}

fn bvncdf(x: f64, y: f64, rho: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let s = (1.0 - rho * rho).sqrt();

    let a = -10.0; // truncate the integral for the standard normal tail
    let b = x;

    let steps = 4000;
    let h = (b - a) / steps as f64;

    let mut sum = 0.0;
    for i in 0..=steps {
        let z = a + i as f64 * h;
        let weight = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        let pdf_z = (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let inner = n.cdf((y - rho * z) / s);
        sum += weight * pdf_z * inner;
    }

    sum * h / 3.0
}

// pub fn option_grid_quote(
//     option: Contract,
//     fundamental: Fundamental,
//     strike: f64,
//     date: DateTime<Utc>,
//     expiration: DateTime<Utc>,
//     price_range: f64,
// ) -> DMatrix<f64> {
//     // uses a numerical solution to Black-Schole with additional noise and microstructure
//     /*
//      * Call: IP = max(S-K, 0)
//      *
//      * Put: IP = max(K-S, 0)
//      *
//      *
//      *
//      *
//      */
//     let price_step = 1.0;
//     let σ = fundamental.standard_deviation;
//     let r = fundamental.bond_rate;
//     let Δ = fundamental.delta; // fractional years
//     let α = (0.5 * σ.powi(2) * Δ) / (1 + r * Δ);
//     let β = (0.5 * r * Δ) / (1 + r * Δ);
//     let γ = 1 / (1 + r * Δ);
//     let a = |i| α * i.powi(2) - β * i;
//     let b = |i| γ - 2 * α * i.powi(2);
//     let c = |i| α * i.powi(2) + β * i;
//     // set up for call for now
//     let time_range = (expiration - date).num_days() as f64 / 365.0;
//     let intrinsic_payoff = SVector::from_fn(price_range as usize, |r| match option {
//         Call => ((r as f64 * price_step) - strike).max(0.0),
//         Put => (strike - (r as f64 * price_step)).max(0.0),
//     });
//     let mut V_grid = DMatrix::<f64>::zeros(price_range, time_range);
//     V_grid.columns_mut(0, time_range - 1);
//     V_grid.set_column(time_range - 1, &intrinsic_payoff);

//     let mut B = SVector::<f64>::zeros(price_range as usize);

//     for j in (time_range / Δ - 1) as usize..0 {
//         let A: DMatrix<f64> =
//             DMatrix::from_fn(price_range as usize, price_range as usize, |r, c| {
//                 if r == c {
//                     b(r as f64)
//                 } else if r == c + 1 {
//                     a(r as f64)
//                 } else if r == c - 1 {
//                     c(r as f64)
//                 } else {
//                     0.0
//                 }
//             });
//         //B = [a_1V,0,j+1, ... , c_N-1V,N,j+1]
//         B[0] = match option {
//             Call => 0.0,
//             Put => a(0) * strike * (-r(time_range - (j + 1)) * time_step).exp(),
//         };
//         B[price_range as usize - 1] = match option {
//             Call => {
//                 price_range
//                     - strike
//                         * (-r(time_range - (j + 1)) * time_step).exp()
//                         * c(price_range as usize - 1)
//             }
//             Put => 0.0,
//         };
//         V_grid.set_column(j - 1, (V_grid.column(j).dot(&A) + B).max(intrinsic_payoff));
//     }
//     V_grid
// }

pub fn bjerksund_stensland_quote(
    // 2002
    option: Contract,
    strike: f64,
    date: DateTime<Utc>,
    expiration: DateTime<Utc>,
    sigma: f64,
    r: f64,
    dividend_yield: f64,
    close: f64,
) -> f64 {
    let σ = sigma;
    let q = dividend_yield;

    let K = match option {
        Contract::Call => strike,
        Contract::Put => close,
    };

    let r = match option {
        Contract::Call => r,
        Contract::Put => q,
    };

    let b: f64 = match option {
        Contract::Call => r - q,
        Contract::Put => q - r,
    };

    let s: f64 = match option {
        //PV(D) = D * e^-rt(until dividend)
        Contract::Call => close, // S* = S - PV(D)
        Contract::Put => strike,
    };

    // Split time for 2 Boundary system
    let t: f64 = (expiration - date).num_days() as f64 / 365.0;
    dbg!(t);
    let t1: f64 = 0.5 * (5.0_f64.sqrt() - 1.0) * t;
    dbg!(t1);

    let β =
        (0.5 - (b / σ.powi(2))) + (((b / σ.powi(2)) - 0.5).powi(2) + 2.0 * r / σ.powi(2)).sqrt();
    dbg!(β);

    let b_infinite = K * β / (β - 1.0);
    dbg!(b_infinite);
    let b_0 = (r * K / (r - b)).max(K);
    dbg!(b_0);

    let h = |t: f64| -(b * t + 2.0 * σ * (t).sqrt()) * (K.powi(2) / ((b_infinite - b_0) * b_0));

    // Boundary system:  X > x > K
    let cx: f64 = b_0 + (b_infinite - b_0) * (1.0 - h(t).exp());
    dbg!(cx);
    let x: f64 = b_0 + (b_infinite - b_0) * (1.0 - h(t - t1).exp());
    dbg!(x);

    if (s >= cx) {
        return s - strike;
    } else {
        let n = Normal::new(0.0, 1.0).unwrap();
        let κ = |γ| (2.0 * b) / σ.powi(2) + (2.0 * γ - 1.0);
        let λ = |γ| -r + γ * b + 0.5 * γ * (γ - 1.0) * σ.powi(2);

        let d1 = |γ: f64| -((s / x).ln() + (b + (γ - 0.5) * σ.powi(2)) * t1) / (σ * t1.sqrt());

        let d2 = |γ: f64| {
            -((cx.powi(2) / (s * x)).ln() + (b + (γ - 0.5) * σ.powi(2)) * t1) / (σ * t1.sqrt())
        };

        let d3 = |γ: f64| -((s / x).ln() - (b + (γ - 0.5) * σ.powi(2)) * t1) / (σ * t1.sqrt());

        let d4 = |γ: f64| {
            -((cx.powi(2) / (s * cx)).ln() - (b + (γ - 0.5) * σ.powi(2)) * t1) / (σ * t1.sqrt())
        };

        let d_1 =
            |γ: f64, h: f64| -((s / h).ln() + (b + (γ - 0.5) * σ.powi(2)) * t) / (σ * t.sqrt());

        let d_2 = |γ: f64, h: f64| {
            -((cx.powi(2) / (s * h)).ln() + (b + (γ - 0.5) * σ.powi(2)) * t) / (σ * t.sqrt())
        };

        let d_3 = |γ: f64, h: f64| {
            -((x.powi(2) / (s * h)).ln() + (b + (γ - 0.5) * σ.powi(2)) * t) / (σ * t.sqrt())
        };

        let d_4 = |γ: f64, h: f64| {
            -(((s * x.powi(2)) / (h * cx.powi(2))).ln() + (b + (γ - 0.5) * σ.powi(2)) * t)
                / (σ * t.sqrt())
        };

        let ϕ = |γ: f64, h: f64| {
            (λ(γ) * t).exp()
                * s.powf(γ)
                * (n.cdf(d_1(γ, h)) - (cx / s).powf(κ(γ)).exp() * n.cdf(d_2(γ, h)))
        };

        let rho = (t1 / t).sqrt();

        let phi = |γ: f64, h: f64| {
            (λ(γ) * t).exp()
                * s.powf(γ)
                * (bvncdf(d1(γ), d_1(γ, h), rho)
                    - (cx / s).powf(κ(γ)) * bvncdf(d2(γ), d_2(γ, h), rho)
                    - (x / s).powf(κ(γ))
                        * (bvncdf(d3(γ), d_3(γ, h), rho)
                            + (x / cx).powf(κ(γ)) * bvncdf(d4(γ), d_4(γ, h), rho)))
        };

        let α = (cx - K) * cx.powf(-β);
        dbg!(α);

        return (α * s.powf(β) - α * ϕ(β, cx) + ϕ(1.0, cx) - ϕ(1.0, x) - K * ϕ(0.0, cx)
            + K * ϕ(0.0, x)
            + α * ϕ(β, x)
            - α * phi(β, x)
            + phi(1.0, x)
            - phi(1.0, K)
            - K * phi(0.0, x)
            + K * phi(0.0, K));
    }
}

// fn fundamental_quote(ticker: Ticker, date: DateTime<Utc>) -> (f64, f64, f64) {
//     let path = "stock";
//     let data = fs::read_to_string(format!("{}/{}/data.json", path, ticker)).unwrap();
//     let stock_data: Vec<Stock> = serde_json::from_str(&data).unwrap();
//     let stock = stock_data.into_iter().find(|s| s.date == date).unwrap();
//     (stock.s, stock.v, stock.r)
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bs_2002() {
        let quote: f64 = bjerksund_stensland_quote(
            Contract::Call,
            115.0,
            Utc::now(),
            Utc::now() + chrono::Duration::days(8),
            0.5444,
            0.035,
            0.0262,
            110.79,
        );
        dbg!(quote);
    }
}
