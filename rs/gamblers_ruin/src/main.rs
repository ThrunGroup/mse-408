use clap::Parser;
use gamblers_ruin::{mcmc, oracle, MCMCParams, MCMCResult, SamplerParams};

/// Gambler's Ruin MCMC
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Lower bound will be "-a"
    #[arg(long, default_value_t = 5)]
    a: i32,
    /// Upper bound will be "b"
    #[arg(long, default_value_t = 5)]
    b: i32,
    /// The starting state; must lie in (-a, b)
    #[arg(long, default_value_t = 0)]
    s0: i32,
    /// The probability of success on each bet.
    #[arg(long, default_value_t = 0.49)]
    p: f64,
    /// Exponentially tilt the distribution using the optimal theta.
    #[arg(long, default_value_t = false)]
    tilt: bool,
    /// The number of burn in samples.
    #[arg(long, default_value_t = 1e5 as usize)]
    n_burn: usize,
    /// The number of threads to use for sampling.
    #[arg(long, default_value_t = 1)]
    n_samplers: usize,
    /// The number of samples between logging the estimate.
    #[arg(long, default_value_t = 1e5 as usize)]
    n_per_log: usize,
    /// When change in the estimate falls below epsilon, stop sampling.
    #[arg(long, default_value_t = 1e-6)]
    epsilon: f64,
}

fn main() {
    let Args {
        a,
        b,
        s0,
        p,
        tilt,
        n_burn,
        n_samplers,
        n_per_log,
        epsilon,
    } = Args::parse();
    let mut theta = 0.0;
    if tilt {
        theta = ((1.0 - p) / p).ln();
    }
    let sampler_params = SamplerParams::new(a, b, s0, p, theta);
    let params =
        MCMCParams::new(n_burn, n_samplers, n_per_log, epsilon, sampler_params);
    let p_hit_b_true = oracle(a, b, s0, p);
    let MCMCResult {
        n_samples, p_hit_b, ..
    } = mcmc(params);
    println!(
        "P(Hit B): {:.6} [Δ {:+.6}] ({} samples)",
        p_hit_b,
        p_hit_b - p_hit_b_true,
        n_samples
    );
}
