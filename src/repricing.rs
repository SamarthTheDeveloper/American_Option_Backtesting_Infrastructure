use rand::thread_rng;
use rand_distr::{Distribution, Poisson, StandardNormal};

pub fn adjust(bid: f64, ask: f64, tick: MPV) -> (f64, f64) {
    let bid_adj = bid - (bid % tick.increment());
    let ask_adj = ask + (tick.increment() - (ask % tick.increment()));
    (bid_adj, ask_adj)
}

/*
 * rho_mean = -0.5
 * phi_rho = 0.95
 * sigma_rho = 0.02
 * rho_iniitial = rho_mean
 *
 * b_mean = 0.15
 * phib = 0.85
 * sigma_b = 0.02
 * b_initial = b_mean
 */
fn ar1_skew_wings(phi: f64, sigma: f64, initial: f64, mean: f64, rho: bool) -> f64 {
    let mut rng = thread_rng();
    let normal = StandardNormal::new().unwrap();
    let x_1 = normal.sample(&mut rng);
    let x_2 = normal.sample(&mut rng);
    let x_3 = normal.sample(&mut rng);
    let c = 0.7;
    let z_p = (c * x_1) + (1 - c.powi(2)).sqrt() * x_2;
    let epsilon = if rho { (z_p * sigma) } else { (x_3 * sigma) };
    let new = (1 - phi) * mean + phi * (initial + epsilon);
    new
}

fn poisson_sample(lambda: f64) -> Result<u64, rand_distr::PoissonError> {
    let mut rng = thread_rng();
    let poisson = Poisson::new(lambda)?;
    let sample: u64 = poisson.sample(&mut rng) as u64;
    Ok(sample)
}
