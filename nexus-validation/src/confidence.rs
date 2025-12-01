//! Confidence interval estimation module
//!
//! Provides various methods for computing confidence intervals including
//! parametric (t-distribution) and non-parametric (bootstrap) approaches.

use crate::{ConfidenceInterval, DescriptiveStats};

/// Bootstrap confidence interval estimator
pub struct BootstrapEstimator {
    /// Number of bootstrap samples
    pub n_samples: usize,
    /// Random seed for reproducibility
    seed: u64,
}

impl BootstrapEstimator {
    /// Create new bootstrap estimator
    pub fn new(n_samples: usize, seed: u64) -> Self {
        Self { n_samples, seed }
    }

    /// Compute bootstrap confidence interval for mean
    pub fn mean_ci(&self, data: &[f64], confidence: f64) -> Option<ConfidenceInterval> {
        if data.is_empty() {
            return None;
        }

        let stats = DescriptiveStats::compute(data)?;
        let n = data.len();

        // Simple LCG for deterministic pseudo-random sampling
        let mut rng_state = self.seed;
        let mut bootstrap_means = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let mut sum = 0.0;
            for _ in 0..n {
                // LCG: x_{n+1} = (a * x_n + c) mod m
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % n;
                sum += data[idx];
            }
            bootstrap_means.push(sum / n as f64);
        }

        // Sort for percentile computation
        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence;
        let lower_idx = ((alpha / 2.0) * self.n_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * self.n_samples as f64) as usize;

        let lower = bootstrap_means.get(lower_idx).copied().unwrap_or(stats.mean);
        let upper = bootstrap_means
            .get(upper_idx.min(bootstrap_means.len() - 1))
            .copied()
            .unwrap_or(stats.mean);

        Some(ConfidenceInterval::new(lower, upper, stats.mean, confidence))
    }

    /// Compute bootstrap confidence interval for median
    pub fn median_ci(&self, data: &[f64], confidence: f64) -> Option<ConfidenceInterval> {
        if data.is_empty() {
            return None;
        }

        let stats = DescriptiveStats::compute(data)?;
        let n = data.len();

        let mut rng_state = self.seed;
        let mut bootstrap_medians = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let mut sample = Vec::with_capacity(n);
            for _ in 0..n {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % n;
                sample.push(data[idx]);
            }
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if n % 2 == 0 {
                (sample[n / 2 - 1] + sample[n / 2]) / 2.0
            } else {
                sample[n / 2]
            };
            bootstrap_medians.push(median);
        }

        bootstrap_medians.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence;
        let lower_idx = ((alpha / 2.0) * self.n_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * self.n_samples as f64) as usize;

        let lower = bootstrap_medians.get(lower_idx).copied().unwrap_or(stats.median);
        let upper = bootstrap_medians
            .get(upper_idx.min(bootstrap_medians.len() - 1))
            .copied()
            .unwrap_or(stats.median);

        Some(ConfidenceInterval::new(lower, upper, stats.median, confidence))
    }

    /// Compute bootstrap confidence interval for ratio of means
    pub fn ratio_ci(
        &self,
        numerator: &[f64],
        denominator: &[f64],
        confidence: f64,
    ) -> Option<ConfidenceInterval> {
        if numerator.is_empty() || denominator.is_empty() {
            return None;
        }

        let num_stats = DescriptiveStats::compute(numerator)?;
        let den_stats = DescriptiveStats::compute(denominator)?;

        if den_stats.mean == 0.0 {
            return None;
        }

        let point_estimate = num_stats.mean / den_stats.mean;
        let n_num = numerator.len();
        let n_den = denominator.len();

        let mut rng_state = self.seed;
        let mut bootstrap_ratios = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let mut num_sum = 0.0;
            let mut den_sum = 0.0;

            for _ in 0..n_num {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % n_num;
                num_sum += numerator[idx];
            }

            for _ in 0..n_den {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % n_den;
                den_sum += denominator[idx];
            }

            if den_sum != 0.0 {
                bootstrap_ratios.push((num_sum / n_num as f64) / (den_sum / n_den as f64));
            }
        }

        bootstrap_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence;
        let lower_idx = ((alpha / 2.0) * bootstrap_ratios.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * bootstrap_ratios.len() as f64) as usize;

        let lower = bootstrap_ratios.get(lower_idx).copied().unwrap_or(point_estimate);
        let upper = bootstrap_ratios
            .get(upper_idx.min(bootstrap_ratios.len().saturating_sub(1)))
            .copied()
            .unwrap_or(point_estimate);

        Some(ConfidenceInterval::new(lower, upper, point_estimate, confidence))
    }
}

/// Bias-corrected and accelerated (BCa) bootstrap
pub struct BcaBootstrap {
    n_samples: usize,
    seed: u64,
}

impl BcaBootstrap {
    pub fn new(n_samples: usize, seed: u64) -> Self {
        Self { n_samples, seed }
    }

    /// Compute BCa confidence interval for mean
    /// 
    /// BCa adjusts for bias and skewness in the bootstrap distribution
    pub fn mean_ci(&self, data: &[f64], confidence: f64) -> Option<ConfidenceInterval> {
        if data.len() < 3 {
            return None;
        }

        let stats = DescriptiveStats::compute(data)?;
        let n = data.len();

        // Generate bootstrap samples
        let mut rng_state = self.seed;
        let mut bootstrap_means = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let mut sum = 0.0;
            for _ in 0..n {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % n;
                sum += data[idx];
            }
            bootstrap_means.push(sum / n as f64);
        }

        // Bias correction factor
        let below_original = bootstrap_means.iter().filter(|&&x| x < stats.mean).count();
        let z0 = inv_normal_cdf(below_original as f64 / self.n_samples as f64);

        // Acceleration factor using jackknife
        let jackknife_means: Vec<f64> = (0..n)
            .map(|i| {
                let sum: f64 = data
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &x)| x)
                    .sum();
                sum / (n - 1) as f64
            })
            .collect();

        let jk_mean: f64 = jackknife_means.iter().sum::<f64>() / n as f64;
        let num: f64 = jackknife_means.iter().map(|&x| (jk_mean - x).powi(3)).sum();
        let den: f64 = jackknife_means.iter().map(|&x| (jk_mean - x).powi(2)).sum();
        let acc = if den != 0.0 { num / (6.0 * den.powf(1.5)) } else { 0.0 };

        // Adjusted percentiles
        let alpha = 1.0 - confidence;
        let z_alpha = inv_normal_cdf(alpha / 2.0);
        let z_1_alpha = inv_normal_cdf(1.0 - alpha / 2.0);

        let p1 = normal_cdf(z0 + (z0 + z_alpha) / (1.0 - acc * (z0 + z_alpha)));
        let p2 = normal_cdf(z0 + (z0 + z_1_alpha) / (1.0 - acc * (z0 + z_1_alpha)));

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = (p1 * self.n_samples as f64) as usize;
        let upper_idx = (p2 * self.n_samples as f64) as usize;

        let lower = bootstrap_means
            .get(lower_idx.min(bootstrap_means.len().saturating_sub(1)))
            .copied()
            .unwrap_or(stats.mean);
        let upper = bootstrap_means
            .get(upper_idx.min(bootstrap_means.len().saturating_sub(1)))
            .copied()
            .unwrap_or(stats.mean);

        Some(ConfidenceInterval::new(lower, upper, stats.mean, confidence))
    }
}

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse normal CDF (probit function) approximation
fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Special case for x = 0
    if x == 0.0 {
        return 0.0;
    }
    
    // Horner's method for the approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_mean_ci() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let estimator = BootstrapEstimator::new(1000, 42);

        let ci = estimator.mean_ci(&data, 0.95).unwrap();

        assert!(ci.lower < ci.point_estimate);
        assert!(ci.upper > ci.point_estimate);
        assert!(ci.contains(ci.point_estimate));
    }

    #[test]
    fn test_bootstrap_median_ci() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let estimator = BootstrapEstimator::new(1000, 42);

        let ci = estimator.median_ci(&data, 0.95).unwrap();

        assert!(ci.lower <= ci.point_estimate);
        assert!(ci.upper >= ci.point_estimate);
    }

    #[test]
    fn test_bootstrap_ratio_ci() {
        let num: Vec<f64> = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let den: Vec<f64> = vec![50.0, 51.0, 49.0, 50.5, 49.5];
        let estimator = BootstrapEstimator::new(1000, 42);

        let ci = estimator.ratio_ci(&num, &den, 0.95).unwrap();

        // Ratio should be approximately 2.0
        assert!(ci.contains(2.0));
    }

    #[test]
    fn test_bca_bootstrap() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let bca = BcaBootstrap::new(1000, 42);

        let ci = bca.mean_ci(&data, 0.95).unwrap();

        assert!(ci.lower < ci.point_estimate);
        assert!(ci.upper > ci.point_estimate);
    }

    #[test]
    fn test_inv_normal_cdf() {
        // Standard normal quantiles
        assert!((inv_normal_cdf(0.5) - 0.0).abs() < 0.01);
        assert!((inv_normal_cdf(0.975) - 1.96).abs() < 0.01);
        assert!((inv_normal_cdf(0.025) - (-1.96)).abs() < 0.01);
    }

    #[test]
    fn test_erf() {
        assert!((erf(0.0) - 0.0).abs() < 1e-10);
        assert!((erf(1.0) - 0.8427).abs() < 0.01);
    }
}
