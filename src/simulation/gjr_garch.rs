//! GJR-GARCH(1,1) - Glosten-Jagannathan-Runkle, with leverage/asymmetry.
//! Written with Claude Modified by me
//!   sigma2_t = omega + (alpha + gamma * 1{eps_{t-1} < 0}) * eps_{t-1}^2 + beta * sigma2_{t-1}
//!   r_t      = mu + eps_t,   eps_t = sigma_t * z_t,   z_t ~ iid (0, 1)
//!
//! gamma > 0 => negative shocks raise volatility more (leverage effect).
//! Stationarity: alpha + beta + gamma/2 < 1.  Long-run var = omega / (1 - alpha - beta - gamma/2).
//!
//! Deps:  rand = "0.8"   rand_distr = "0.4"

use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Copy, Debug)]
pub struct GjrGarch {
    pub omega: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub beta: f64,
    pub mu: f64,
}

impl GjrGarch {
    pub fn new(omega: f64, alpha: f64, gamma: f64, beta: f64, mu: f64) -> Self {
        Self {
            omega,
            alpha,
            gamma,
            beta,
            mu,
        }
    }
    pub fn uncond_var(&self) -> Option<f64> {
        let p = self.alpha + self.beta + 0.5 * self.gamma;
        (p < 1.0).then(|| self.omega / (1.0 - p))
    }
    #[inline]
    pub fn next_var(&self, prev_eps: f64, prev_s2: f64) -> f64 {
        let lev = if prev_eps < 0.0 { self.gamma } else { 0.0 };
        self.omega + (self.alpha + lev) * prev_eps * prev_eps + self.beta * prev_s2
    }
    pub fn seed_var(&self) -> f64 {
        self.uncond_var()
            .unwrap_or_else(|| self.omega / (1.0 - self.beta).max(1e-8))
    }
    pub fn simulate<R, F>(&self, n: usize, rng: &mut R, mut z: F) -> Vec<f64>
    where
        R: Rng + ?Sized,
        F: FnMut(&mut R) -> f64,
    {
        let mut returns = Vec::with_capacity(n);
        let mut s2 = self.seed_var();
        let mut prev_eps = 0.0_f64;
        for t in 0..n {
            if t > 0 {
                s2 = self.next_var(prev_eps, s2);
                /* generally we can derive a sigma/standard deviation value
                    however, typically we want our sigma to be calibrated at the year basis so
                    sqrt(sigma * ppy); therefore, its better to not store it
                */
            }
            let eps = s2.sqrt() * z(rng);
            returns.push(self.mu + eps);
            prev_eps = eps;
        }
        returns
    }

    pub fn simulate_normal<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Vec<f64> {
        let d = Normal::new(0.0, 1.0).unwrap();
        self.simulate(n, rng, |r| d.sample(r))
    }

    pub fn neg_loglik(&self, r: &[f64]) -> f64 {
        if r.len() < 2 {
            return f64::INFINITY;
        }
        let mut s2 = self.seed_var();
        let mut prev_eps = r[0] - self.mu;
        let mut nll = 0.0;
        for &ri in &r[1..] {
            s2 = self.next_var(prev_eps, s2);
            if s2 <= 0.0 || !s2.is_finite() {
                return f64::INFINITY;
            }
            let eps = ri - self.mu;
            nll += 0.5 * ((std::f64::consts::TAU * s2).ln() + eps * eps / s2);
            prev_eps = eps;
        }
        nll
    }
    pub fn fit(r: &[f64]) -> Self {
        let n = r.len() as f64;
        let mean = r.iter().sum::<f64>() / n;
        let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let unpack = |x: &[f64]| -> GjrGarch {
            let omega = x[0].exp();
            let mu = x[1];
            let p = sigmoid(x[2]);
            let (sa, sb, sg) = softmax3(x[3], x[4], x[5]);
            GjrGarch::new(omega, sa * p, 2.0 * sg * p, sb * p, mu)
        };
        let obj = |x: &[f64]| unpack(x).neg_loglik(r);
        let p0 = 0.95;
        let base = vec![
            (var * (1.0 - p0)).max(1e-12).ln(),
            mean,
            logit(p0),
            0.05_f64.ln(),
            0.85_f64.ln(),
            0.10_f64.ln(),
        ];
        let mut best = base.clone();
        let mut best_f = obj(&best);
        for seed_p in [0.90, 0.97, 0.80] {
            let mut start = base.clone();
            start[2] = logit(seed_p);
            let sol = nelder_mead(&obj, start, 4000, 1e-10);
            let f = obj(&sol);
            if f < best_f {
                best_f = f;
                best = sol;
            }
        }
        unpack(&best)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}
fn softmax3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let m = a.max(b).max(c);
    let (ea, eb, ec) = ((a - m).exp(), (b - m).exp(), (c - m).exp());
    let s = ea + eb + ec;
    (ea / s, eb / s, ec / s)
}

fn nelder_mead<F: Fn(&[f64]) -> f64>(f: &F, x0: Vec<f64>, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut v = x0.clone();
        let step = if v[i].abs() > 1e-8 { 0.05 * v[i] } else { 0.05 };
        v[i] += step;
        simplex.push(v);
    }
    let mut fx: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let (refl, exp, con, shr) = (1.0, 2.0, 0.5, 0.5);
    for _ in 0..max_iter {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&i, &j| fx[i].partial_cmp(&fx[j]).unwrap());
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        fx = idx.iter().map(|&i| fx[i]).collect();
        if (fx[n] - fx[0]).abs() <= tol * (fx[0].abs() + tol) {
            break;
        }
        let mut cen = vec![0.0; n];
        for v in simplex.iter().take(n) {
            for k in 0..n {
                cen[k] += v[k] / n as f64;
            }
        }
        let worst = simplex[n].clone();
        let reflect: Vec<f64> = (0..n)
            .map(|k| cen[k] + refl * (cen[k] - worst[k]))
            .collect();
        let fr = f(&reflect);
        if fr < fx[0] {
            let expand: Vec<f64> = (0..n).map(|k| cen[k] + exp * (cen[k] - worst[k])).collect();
            let fe = f(&expand);
            if fe < fr {
                simplex[n] = expand;
                fx[n] = fe;
            } else {
                simplex[n] = reflect;
                fx[n] = fr;
            }
        } else if fr < fx[n - 1] {
            simplex[n] = reflect;
            fx[n] = fr;
        } else {
            let contract: Vec<f64> = (0..n).map(|k| cen[k] + con * (worst[k] - cen[k])).collect();
            let fc = f(&contract);
            if fc < fx[n] {
                simplex[n] = contract;
                fx[n] = fc;
            } else {
                let best = simplex[0].clone();
                for i in 1..=n {
                    for k in 0..n {
                        simplex[i][k] = best[k] + shr * (simplex[i][k] - best[k]);
                    }
                    fx[i] = f(&simplex[i]);
                }
            }
        }
    }
    let mut bi = 0;
    for i in 1..=n {
        if fx[i] < fx[bi] {
            bi = i;
        }
    }
    simplex[bi].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_parameters() {
        let truth = GjrGarch::new(2.0e-6, 0.04, 0.10, 0.88, 0.0003);
        let mut rng = rand::rng();
        let rets = truth.simulate_normal(8000, &mut rng);
        let fit = GjrGarch::fit(&rets);

        // persistence and total ARCH effect are well identified;
        // the alpha-vs-gamma split is looser (partial collinearity on down-days).
        let truth_p = truth.alpha + truth.beta + 0.5 * truth.gamma;
        let fit_p = fit.alpha + fit.beta + 0.5 * fit.gamma;
        assert!(
            (fit_p - truth_p).abs() < 0.02,
            "persistence off: {fit_p} vs {truth_p}"
        );
        assert!(fit.gamma > 0.0, "leverage should be positive");
        // fitted is the in-sample MLE, so its NLL must not exceed the truth's
        assert!(fit.neg_loglik(&rets) <= truth.neg_loglik(&rets) + 1e-6);
    }
}
