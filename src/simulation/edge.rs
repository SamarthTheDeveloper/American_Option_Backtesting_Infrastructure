//! EDGE — Efficient Discrete Generalized Estimator of the bid-ask spread.
//! Written with the help nClaude
//! Implements, exactly as derived in the paper:
//!   Ardia, Guidotti & Kroencke (2024),
//!   "Efficient Estimation of Bid-Ask Spreads from Open, High, Low, and Close
//!   Prices", Journal of Financial Economics 161.
//!
//! Equation references below (Eq. 7, 9, 19–26, Table 1) are to that paper.
//!
//! Inputs: raw (NOT log) open/high/low/close price series in chronological
//! order, one bar per period (any frequency: daily, hourly, minute).
//! Output: the root mean squared effective spread S over the sample, as a
//! fraction of price (0.01 = 1%). One estimate per call — resolution is the
//! window you pass in.

/// EDGE estimator (Eq. 23–25), with the non-negativity treatment of Eq. (26)
/// applied to the final estimate when `sign == false`.
///
/// * `open`, `high`, `low`, `close` — equal-length raw price series.
/// * `sign` — if `false` (paper default), negative squared-spread estimates
///   are truncated to zero: S = sqrt(max(0, S²)).
///   If `true`, returns the signed estimate S = sign(S²)·sqrt(|S²|)
///   (Internet Appendix Eq. I.1), useful to avoid zero-inflation.
///
/// Returns `None` when the estimator is undefined: fewer than 3 bars,
/// non-finite prices, or degenerate denominators (e.g. the instrument never
/// trades away from the previous close, so E[τ] = 0 or π is undefined) —
/// the cases the paper treats as missing rather than imputing zero.
pub fn edge(open: &[f64], high: &[f64], low: &[f64], close: &[f64], sign: bool) -> Option<f64> {
    let n = open.len();
    if high.len() != n || low.len() != n || close.len() != n {
        return None;
    }
    // The covariance needs at least two returns => at least 3 observations
    // (see Section 4.1 of the paper).
    if n < 3 {
        return None;
    }
    if !open
        .iter()
        .chain(high)
        .chain(low)
        .chain(close)
        .all(|p| p.is_finite() && *p > 0.0)
    {
        return None;
    }

    // Log-prices: p = log(P), Eq. (4). Mid-price eta = (h + l)/2 (Table 1).
    let o: Vec<f64> = open.iter().map(|p| p.ln()).collect();
    let h: Vec<f64> = high.iter().map(|p| p.ln()).collect();
    let l: Vec<f64> = low.iter().map(|p| p.ln()).collect();
    let c: Vec<f64> = close.iter().map(|p| p.ln()).collect();
    let eta: Vec<f64> = h.iter().zip(&l).map(|(hi, li)| 0.5 * (hi + li)).collect();

    let m = (n - 1) as f64; // number of usable periods t = 1..n-1 (lag of 1)

    // ---------------------------------------------------------------------
    // Pass 1: indicator tau (Eq. 7), the joint probabilities entering the
    // pi coefficients (Table 1), and the means needed to de-mean returns
    // (Eq. 9).
    // ---------------------------------------------------------------------
    let mut tau_sum = 0.0; // E[tau] * m
    let mut cnt_o_ne_h = 0.0; // P[o_t != h_t,     tau_t = 1] * m
    let mut cnt_o_ne_l = 0.0; // P[o_t != l_t,     tau_t = 1] * m
    let mut cnt_c_ne_h1 = 0.0; // P[c_{t-1} != h_{t-1}, tau_t = 1] * m
    let mut cnt_c_ne_l1 = 0.0; // P[c_{t-1} != l_{t-1}, tau_t = 1] * m

    // Sums of the five raw returns used by the moment conditions:
    // r1 = eta_t - o_t        (open-to-mid)
    // r2 = o_t   - eta_{t-1}  (lagged-mid-to-open)
    // r3 = eta_t - c_{t-1}    (close-to-mid)
    // r4 = c_{t-1} - eta_{t-1} (lagged-mid-to-close)
    // r5 = o_t   - c_{t-1}    (close-to-open)
    let mut sum_r = [0.0f64; 5];

    for t in 1..n {
        // Eq. (7): tau_t = 0 iff h_t = l_t = c_{t-1}. Compare raw prices for
        // exact equality (forward-filled / no-trade bars replicate them
        // exactly, so this is an exact comparison, not a tolerance test).
        let tau = if high[t] == low[t] && low[t] == close[t - 1] {
            0.0
        } else {
            1.0
        };
        tau_sum += tau;

        if tau == 1.0 {
            if open[t] != high[t] {
                cnt_o_ne_h += 1.0;
            }
            if open[t] != low[t] {
                cnt_o_ne_l += 1.0;
            }
            if close[t - 1] != high[t - 1] {
                cnt_c_ne_h1 += 1.0;
            }
            if close[t - 1] != low[t - 1] {
                cnt_c_ne_l1 += 1.0;
            }
        }

        sum_r[0] += eta[t] - o[t];
        sum_r[1] += o[t] - eta[t - 1];
        sum_r[2] += eta[t] - c[t - 1];
        sum_r[3] += c[t - 1] - eta[t - 1];
        sum_r[4] += o[t] - c[t - 1];
    }

    // E[tau] = 0 means prices never move off the previous close: undefined.
    if tau_sum == 0.0 {
        return None;
    }

    // Table 1 coefficients:
    //   pi_o = -8 / ( P[o != h, tau=1] + P[o != l, tau=1] )
    //   pi_c = -8 / ( P[c1 != h1, tau=1] + P[c1 != l1, tau=1] )
    // Probabilities are sample frequencies (counts / m).
    let p_o = (cnt_o_ne_h + cnt_o_ne_l) / m;
    let p_c = (cnt_c_ne_h1 + cnt_c_ne_l1) / m;
    if p_o == 0.0 || p_c == 0.0 {
        return None; // denominator of pi undefined (Section 4.1)
    }
    let pi_o = -8.0 / p_o;
    let pi_c = -8.0 / p_c;

    // Eq. (9) de-meaning constants: rbar_t = r_t - tau_t * E[r]/E[tau].
    let d: Vec<f64> = sum_r.iter().map(|s| s / tau_sum).collect();

    // ---------------------------------------------------------------------
    // Pass 2: moment series x1, x2 (Eq. 24) built from de-meaned returns,
    // then their sample means and variances for the GMM weights (Eq. 25)
    // with the diagonal weighting matrix used for EDGE (Eq. 23).
    // ---------------------------------------------------------------------
    let (mut s1, mut s2) = (0.0f64, 0.0f64); // sum of x1, x2
    let (mut q1, mut q2) = (0.0f64, 0.0f64); // sum of x1^2, x2^2

    for t in 1..n {
        let tau = if high[t] == low[t] && low[t] == close[t - 1] {
            0.0
        } else {
            1.0
        };

        let r1 = (eta[t] - o[t]) - tau * d[0];
        let r2 = (o[t] - eta[t - 1]) - tau * d[1];
        let r3 = (eta[t] - c[t - 1]) - tau * d[2];
        let r4 = (c[t - 1] - eta[t - 1]) - tau * d[3];
        let r5 = (o[t] - c[t - 1]) - tau * d[4];

        // Eq. (24):
        // x1 = (pi_o/2)(eta_t - o_t)(o_t - eta_{t-1})
        //    + (pi_c/2)(eta_t - c_{t-1})(c_{t-1} - eta_{t-1})
        // x2 = (pi_o/2)(eta_t - o_t)(o_t - c_{t-1})
        //    + (pi_c/2)(o_t - c_{t-1})(c_{t-1} - eta_{t-1})
        let x1 = 0.5 * (pi_o * r1 * r2 + pi_c * r3 * r4);
        let x2 = 0.5 * (pi_o * r1 * r5 + pi_c * r5 * r4);

        s1 += x1;
        s2 += x2;
        q1 += x1 * x1;
        q2 += x2 * x2;
    }

    let mu1 = s1 / m; // E[x1]
    let mu2 = s2 / m; // E[x2]
    let v1 = q1 / m - mu1 * mu1; // Var[x1] (population moments, as in Eq. 25)
    let v2 = q2 / m - mu2 * mu2; // Var[x2]

    // Eq. (25): inverse-variance weights from the diagonal covariance matrix.
    // If both variances vanish (constant moments), fall back to equal weights.
    let s2_edge = if v1 + v2 > 0.0 {
        let w1 = v2 / (v1 + v2);
        let w2 = v1 / (v1 + v2);
        w1 * mu1 + w2 * mu2 // Eq. (23): S^2 = w1 E[x1] + w2 E[x2]
    } else {
        0.5 * (mu1 + mu2)
    };

    // Eq. (26) or Internet Appendix Eq. (I.1).
    let s = if sign {
        s2_edge.signum() * s2_edge.abs().sqrt()
    } else {
        s2_edge.max(0.0).sqrt()
    };
    Some(s)
}

// =========================================================================
// Validation against the paper's own Monte Carlo design (Section 3.1):
// minute-level GBM fundamental, daily sigma = 3%, trade prices = fundamental
// * (1 ± S/2) with 50/50 bid/ask, bars forward-filled when no trade occurs.
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift RNG so the tests need no external crates.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal via Box-Muller.
        fn gauss(&mut self) -> f64 {
            let (u1, u2) = (self.uniform().max(1e-15), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// Simulate `days` daily OHLC bars per Section 3.1 of the paper.
    /// `spread` is the assumed effective spread, `p_trade` the per-minute
    /// probability of observing a trade.
    fn simulate(
        days: usize,
        spread: f64,
        p_trade: f64,
        rng: &mut Rng,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let sigma_min = 0.03 / (390.0f64).sqrt();
        let mut fund = 1.0f64;
        let mut prev_close = 1.0f64;
        let (mut o, mut h, mut l, mut c) = (vec![], vec![], vec![], vec![]);

        for _ in 0..days {
            let mut day: Vec<f64> = Vec::new();
            for _ in 0..390 {
                fund *= (sigma_min * rng.gauss()).exp();
                if rng.uniform() < p_trade {
                    let side = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                    day.push(fund * (1.0 + side * spread / 2.0));
                }
            }
            if day.is_empty() {
                // No trades: forward-fill OHLC with the previous close.
                o.push(prev_close);
                h.push(prev_close);
                l.push(prev_close);
                c.push(prev_close);
            } else {
                o.push(day[0]);
                h.push(day.iter().cloned().fold(f64::MIN, f64::max));
                l.push(day.iter().cloned().fold(f64::MAX, f64::min));
                c.push(*day.last().unwrap());
                prev_close = *day.last().unwrap();
            }
        }
        (o, h, l, c)
    }

    #[test]
    fn unbiased_frequent_trading() {
        // 390 trades/day, S = 1%: long-sample estimate should sit on 1%.
        let mut rng = Rng(88172645463325252);
        let (o, h, l, c) = simulate(2000, 0.01, 1.0, &mut rng);
        let s = edge(&o, &h, &l, &c, false).unwrap();
        assert!((s - 0.01).abs() < 0.0015, "S = {s}");
    }

    #[test]
    fn unbiased_infrequent_trading() {
        // ~4 trades/day: CS/AR collapse toward zero here (Fig. 2); EDGE must not.
        let mut rng = Rng(1181783497276652981);
        let (o, h, l, c) = simulate(4000, 0.01, 1.0 / 100.0, &mut rng);
        let s = edge(&o, &h, &l, &c, false).unwrap();
        assert!((s - 0.01).abs() < 0.003, "S = {s}");
    }

    #[test]
    fn large_spread() {
        let mut rng = Rng(2862933555777941757);
        let (o, h, l, c) = simulate(1000, 0.03, 1.0, &mut rng);
        let s = edge(&o, &h, &l, &c, false).unwrap();
        assert!((s - 0.03).abs() < 0.003, "S = {s}");
    }

    #[test]
    fn undefined_cases() {
        // Too short.
        assert!(edge(&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], false).is_none());
        // Constant prices: tau = 0 everywhere => undefined, not zero.
        let p = vec![1.0; 30];
        assert!(edge(&p, &p, &p, &p, false).is_none());
    }

    /// Reproduces the qualitative content of Figure 2: EDGE stays unbiased
    /// as trading frequency falls. Run with `-- --nocapture` to see values.
    #[test]
    fn report_across_trading_frequencies() {
        for (label, spread, p) in [
            ("S=1.00%, 390 trades/day", 0.01, 1.0),
            ("S=1.00%,  ~39 trades/day", 0.01, 0.1),
            ("S=1.00%,   ~4 trades/day", 0.01, 0.01),
            ("S=3.00%, 390 trades/day", 0.03, 1.0),
        ] {
            let mut rng = Rng(0x9E3779B97F4A7C15);
            let (o, h, l, c) = simulate(4000, spread, p, &mut rng);
            let s = edge(&o, &h, &l, &c, false).unwrap();
            println!("{label}: EDGE = {:.4}%", s * 100.0);
            assert!((s - spread).abs() < 0.15 * spread, "{label}: S = {s}");
        }
    }
}
