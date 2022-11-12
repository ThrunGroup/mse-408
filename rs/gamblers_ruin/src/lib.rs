use rand::distributions::{Bernoulli, Distribution};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

// TODO:
// 1. How do we phrase this as a learning problem
// 2. What is the functional form of u(x)
// 3. Variance of exponential tilting? -> relative error

#[derive(Clone, Debug)]
struct Task {
    tx: mpsc::Sender<Sample>,
    stop: Arc<AtomicBool>,
    sampler_params: SamplerParams,
}

#[derive(Clone, Debug)]
struct Sample {
    hit_b: bool,
    trajectory: Vec<i32>,
    weight: f64,
}

#[derive(Clone, Debug)]
pub struct SamplerParams {
    a: i32,
    b: i32,
    s0: i32,
    p: f64,
    theta: f64,
}

#[derive(Clone, Debug)]
pub struct MCMCParams {
    n_burn: usize,
    n_samplers: usize,
    n_per_log: usize,
    epsilon: f64,
    sampler_params: SamplerParams,
}

#[derive(Clone, Debug)]
pub struct MCMCResult {
    pub n_samples: usize,
    pub p_hit_b: f64,
    pub t_cond_b: Vec<Vec<i32>>,
}

impl SamplerParams {
    pub fn new(a: i32, b: i32, s0: i32, p: f64, theta: f64) -> Self {
        assert!(-a < b, "-a must be less than b!");
        assert!(-a < s0 && s0 < b, "s0 must lie in (-a, b)!");
        assert!(p > 0.0 && p < 1.0, "p must lie in (0,1)!");
        assert!(theta >= 0.0, "theta must be greater than 0!");
        Self { a, b, s0, p, theta }
    }
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            a: 5,
            b: 5,
            s0: 0,
            p: 0.49,
            theta: 0.0,
        }
    }
}

impl MCMCParams {
    pub fn new(
        n_burn: usize,
        n_samplers: usize,
        n_per_log: usize,
        epsilon: f64,
        sampler_params: SamplerParams,
    ) -> Self {
        assert!(n_burn > 0, "n_burn must be greater than 0!");
        assert!(n_samplers > 0, "n_samplers must be greater than 0!");
        assert!(n_per_log > 0, "n_per_log must be greater than 0!");
        assert!(epsilon > 0. && epsilon < 1., "epsilon must lie in (0, 1)!");
        Self {
            n_burn,
            n_samplers,
            n_per_log,
            epsilon,
            sampler_params,
        }
    }
}

impl Default for MCMCParams {
    fn default() -> Self {
        Self {
            n_burn: 1e3 as usize,
            n_samplers: 1,
            n_per_log: 1e5 as usize,
            epsilon: 1e-6,
            sampler_params: SamplerParams::default(),
        }
    }
}

pub fn oracle(a: i32, b: i32, s0: i32, p: f64) -> f64 {
    if p == 0.5 {
        return (s0 / (b + a)) as f64;
    }
    let q = 1.0 - p;
    (1.0 - f64::powi(q / p, b)) / (1.0 - f64::powi(q / p, b + a))
}

pub fn mcmc(params: MCMCParams) -> MCMCResult {
    let MCMCParams {
        n_burn,
        n_samplers,
        n_per_log,
        epsilon,
        sampler_params: SamplerParams { a, b, s0, p, theta },
    } = params;
    let (tx, rx) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let task = Task {
        tx: tx.clone(),
        stop: stop.clone(),
        sampler_params: SamplerParams { a, b, s0, p, theta },
    };
    let mut samplers = Vec::with_capacity(n_samplers);
    for _ in 0..n_samplers {
        let t = task.clone();
        samplers.push(thread::spawn(move || sampler(t)));
    }
    let mut n_samples: usize = 0;
    let mut delta: f64 = 0.0;
    let mut p_hit_b: f64 = 0.0;
    let mut t_cond_b: Vec<Vec<i32>> = Vec::new();
    let p_hit_b_true = oracle(a, b, s0, p);
    while n_samples < n_burn || delta.abs() > epsilon {
        let sample = rx.recv().unwrap();
        let p_hat = (sample.hit_b as i64) as f64 * sample.weight;
        n_samples += 1;
        delta = (p_hat - p_hit_b) / n_samples as f64;
        p_hit_b += delta;
        if sample.hit_b {
            t_cond_b.push(sample.trajectory);
        }
        if n_samples % n_per_log == 0 {
            println!(
                "P(Hit B): {:.6} [Δ {:+.6}] ({} samples)",
                p_hit_b,
                p_hit_b - p_hit_b_true,
                n_samples
            );
        }
    }
    stop.store(true, Ordering::Relaxed);
    for sampler in samplers {
        sampler.join().unwrap();
    }
    MCMCResult {
        n_samples,
        p_hit_b,
        t_cond_b,
    }
}

fn sampler(task: Task) {
    let Task {
        tx,
        stop,
        sampler_params: SamplerParams { a, b, s0, p, theta },
    } = task;
    let (q, e_theta, e_neg_theta) = (1.0 - p, theta.exp(), (-theta).exp());
    let p_tilt = p * e_theta / (q * e_neg_theta + p * e_theta);
    let psi_theta = (q * e_neg_theta + p * e_theta).ln();
    let d = Bernoulli::new(p_tilt).unwrap();
    let mut w = 1.0;
    let mut rng = rand::thread_rng();
    while !stop.load(Ordering::Relaxed) {
        let mut t: Vec<i32> = vec![s0];
        let mut t_last = s0;
        let mut y: i32;
        let mut x: i32;
        while t_last != -a && t_last != b {
            y = d.sample(&mut rng).into();
            x = 2 * y - 1;
            t_last += x;
            t.push(t_last);
        }
        let s = (t_last - s0) as f64;
        let n = t.len() as f64;
        if theta > 0.0 {
            w = 1.0 / (theta * s - n * psi_theta).exp();
        };
        tx.send(Sample {
            hit_b: t_last == b,
            trajectory: t,
            weight: w,
        })
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_sampler() {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let sampler_params = SamplerParams::default();
        let task = Task {
            tx,
            stop: stop.clone(),
            sampler_params,
        };
        let handle = thread::spawn(move || sampler(task));
        let sample = rx.recv().unwrap();
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        assert!(sample.trajectory.len() > 1, "Invalid sample trajectory!");
    }
}
