//! Statistical Validation Framework for Nexus Memory System
//!
//! This module provides comprehensive statistical analysis for validating
//! performance claims and ensuring reproducibility of benchmarks.
//!
//! # Features
//!
//! - Hypothesis testing with configurable significance levels
//! - Confidence interval estimation (parametric and bootstrap)
//! - Effect size analysis (Cohen's d, Cliff's delta)
//! - Distribution fitting and goodness-of-fit tests
//! - Outlier detection and robust statistics

pub mod confidence;
pub mod statistical;

use std::fmt;

/// Statistical validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the validation passed
    pub passed: bool,
    /// Confidence level used
    pub confidence_level: f64,
    /// Computed p-value (if applicable)
    pub p_value: Option<f64>,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
    /// Description of the result
    pub description: String,
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ValidationResult {{ passed: {}, confidence: {:.2}%, p_value: {:?}, effect_size: {:?} }}",
            self.passed,
            self.confidence_level * 100.0,
            self.p_value,
            self.effect_size
        )
    }
}

/// Configuration for statistical analysis
#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    /// Significance level (alpha) for hypothesis tests
    pub alpha: f64,
    /// Confidence level for intervals (1 - alpha)
    pub confidence_level: f64,
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Minimum sample size for parametric tests
    pub min_sample_size: usize,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            confidence_level: 0.95,
            bootstrap_samples: 10000,
            seed: Some(42),
            min_sample_size: 30,
        }
    }
}

/// Descriptive statistics for a sample
#[derive(Debug, Clone)]
pub struct DescriptiveStats {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl DescriptiveStats {
    /// Compute descriptive statistics for a sample
    pub fn compute(data: &[f64]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let n = data.len();
        let count = n;

        // Mean
        let sum: f64 = data.iter().sum();
        let mean = sum / n as f64;

        // Variance and std dev
        let variance = if n > 1 {
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        // Min/Max
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Sorted for quantiles
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Quantiles
        let median = percentile(&sorted, 50.0);
        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        // Skewness (Fisher-Pearson)
        let skewness = if std_dev > 0.0 && n > 2 {
            let m3: f64 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;
            m3 / std_dev.powi(3)
        } else {
            0.0
        };

        // Kurtosis (excess)
        let kurtosis = if std_dev > 0.0 && n > 3 {
            let m4: f64 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n as f64;
            m4 / std_dev.powi(4) - 3.0
        } else {
            0.0
        };

        Some(Self {
            count,
            mean,
            std_dev,
            variance,
            min,
            max,
            median,
            q1,
            q3,
            iqr,
            skewness,
            kurtosis,
        })
    }
}

impl fmt::Display for DescriptiveStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Descriptive Statistics:")?;
        writeln!(f, "  N:        {}", self.count)?;
        writeln!(f, "  Mean:     {:.4}", self.mean)?;
        writeln!(f, "  Std Dev:  {:.4}", self.std_dev)?;
        writeln!(f, "  Min:      {:.4}", self.min)?;
        writeln!(f, "  Q1:       {:.4}", self.q1)?;
        writeln!(f, "  Median:   {:.4}", self.median)?;
        writeln!(f, "  Q3:       {:.4}", self.q3)?;
        writeln!(f, "  Max:      {:.4}", self.max)?;
        writeln!(f, "  Skewness: {:.4}", self.skewness)?;
        writeln!(f, "  Kurtosis: {:.4}", self.kurtosis)
    }
}

/// Calculate percentile using linear interpolation
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let n = sorted.len();
    let rank = (p / 100.0) * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper || upper >= n {
        sorted[lower.min(n - 1)]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Confidence interval
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub point_estimate: f64,
    pub confidence_level: f64,
}

impl ConfidenceInterval {
    /// Create a new confidence interval
    pub fn new(lower: f64, upper: f64, point: f64, level: f64) -> Self {
        Self {
            lower,
            upper,
            point_estimate: point,
            confidence_level: level,
        }
    }

    /// Check if a value is within the interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Width of the interval
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Compute t-distribution based CI for mean
    pub fn for_mean(data: &[f64], confidence: f64) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }

        let stats = DescriptiveStats::compute(data)?;
        let n = data.len() as f64;
        let se = stats.std_dev / n.sqrt();

        // Use approximation for t-critical value
        // For n > 30, t approaches z
        let alpha = 1.0 - confidence;
        let t_crit = if data.len() > 30 {
            z_critical(1.0 - alpha / 2.0)
        } else {
            t_critical(data.len() - 1, 1.0 - alpha / 2.0)
        };

        let margin = t_crit * se;
        Some(Self::new(
            stats.mean - margin,
            stats.mean + margin,
            stats.mean,
            confidence,
        ))
    }
}

impl fmt::Display for ConfidenceInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.4} [{:.4}, {:.4}] ({:.0}% CI)",
            self.point_estimate,
            self.lower,
            self.upper,
            self.confidence_level * 100.0
        )
    }
}

/// Approximate z-critical value for standard normal
fn z_critical(p: f64) -> f64 {
    // Approximation using rational function
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    let p = if p > 0.5 { 1.0 - p } else { p };
    let sign = if p > 0.5 { -1.0 } else { 1.0 };

    let t = (-2.0 * p.ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    sign * (t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
}

/// Approximate t-critical value
fn t_critical(df: usize, p: f64) -> f64 {
    // For large df, t approaches z
    if df > 100 {
        return z_critical(p);
    }

    // Simple approximation based on normal with correction
    let z = z_critical(p);
    let df_f = df as f64;

    // Cornish-Fisher expansion correction
    let g1 = (z.powi(3) + z) / (4.0 * df_f);
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df_f * df_f);

    z + g1 + g2
}

/// Hypothesis test result
#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom (if applicable)
    pub df: Option<f64>,
    /// Whether to reject null hypothesis
    pub reject_null: bool,
    /// Significance level used
    pub alpha: f64,
    /// Description of the test
    pub test_name: String,
}

impl fmt::Display for HypothesisTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "  Statistic: {:.4}", self.statistic)?;
        writeln!(f, "  P-value:   {:.6}", self.p_value)?;
        if let Some(df) = self.df {
            writeln!(f, "  DF:        {:.1}", df)?;
        }
        writeln!(f, "  Alpha:     {:.4}", self.alpha)?;
        writeln!(
            f,
            "  Decision:  {} null hypothesis",
            if self.reject_null { "Reject" } else { "Fail to reject" }
        )
    }
}

/// Two-sample t-test (Welch's)
pub fn welch_t_test(sample1: &[f64], sample2: &[f64], alpha: f64) -> Option<HypothesisTestResult> {
    let stats1 = DescriptiveStats::compute(sample1)?;
    let stats2 = DescriptiveStats::compute(sample2)?;

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    // Pooled standard error
    let se = ((stats1.variance / n1) + (stats2.variance / n2)).sqrt();
    if se == 0.0 {
        return None;
    }

    // t-statistic
    let t = (stats1.mean - stats2.mean) / se;

    // Welch-Satterthwaite degrees of freedom
    let v1 = stats1.variance / n1;
    let v2 = stats2.variance / n2;
    let df = (v1 + v2).powi(2) / (v1.powi(2) / (n1 - 1.0) + v2.powi(2) / (n2 - 1.0));

    // P-value (two-tailed) using approximation
    let p_value = 2.0 * (1.0 - t_cdf(t.abs(), df));

    Some(HypothesisTestResult {
        statistic: t,
        p_value,
        df: Some(df),
        reject_null: p_value < alpha,
        alpha,
        test_name: "Welch's Two-Sample t-Test".to_string(),
    })
}

/// Approximate t-distribution CDF
fn t_cdf(t: f64, df: f64) -> f64 {
    // Approximation using regularized incomplete beta function
    // For simplicity, use normal approximation for large df
    if df > 30.0 {
        normal_cdf(t)
    } else {
        // Simple approximation
        let x = df / (df + t * t);
        0.5 + 0.5 * (1.0 - x.powf(df / 2.0)).copysign(t)
    }
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Effect size: Cohen's d
pub fn cohens_d(sample1: &[f64], sample2: &[f64]) -> Option<f64> {
    let stats1 = DescriptiveStats::compute(sample1)?;
    let stats2 = DescriptiveStats::compute(sample2)?;

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    // Pooled standard deviation
    let pooled_var =
        ((n1 - 1.0) * stats1.variance + (n2 - 1.0) * stats2.variance) / (n1 + n2 - 2.0);
    let pooled_sd = pooled_var.sqrt();

    if pooled_sd == 0.0 {
        return None;
    }

    Some((stats1.mean - stats2.mean) / pooled_sd)
}

/// Effect size interpretation
pub fn interpret_cohens_d(d: f64) -> &'static str {
    let d = d.abs();
    if d < 0.2 {
        "negligible"
    } else if d < 0.5 {
        "small"
    } else if d < 0.8 {
        "medium"
    } else {
        "large"
    }
}

/// Benchmark comparison result
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub baseline_stats: DescriptiveStats,
    pub treatment_stats: DescriptiveStats,
    pub difference: f64,
    pub percent_change: f64,
    pub confidence_interval: ConfidenceInterval,
    pub test_result: HypothesisTestResult,
    pub effect_size: f64,
    pub effect_interpretation: String,
    pub is_improvement: bool,
    pub is_significant: bool,
}

impl BenchmarkComparison {
    /// Compare two benchmark samples
    pub fn compare(
        baseline: &[f64],
        treatment: &[f64],
        config: &StatisticalConfig,
    ) -> Option<Self> {
        let baseline_stats = DescriptiveStats::compute(baseline)?;
        let treatment_stats = DescriptiveStats::compute(treatment)?;

        let difference = treatment_stats.mean - baseline_stats.mean;
        let percent_change = if baseline_stats.mean != 0.0 {
            (difference / baseline_stats.mean) * 100.0
        } else {
            0.0
        };

        // Confidence interval for difference
        let combined: Vec<f64> = baseline
            .iter()
            .chain(treatment.iter())
            .cloned()
            .collect();
        let ci = ConfidenceInterval::for_mean(&combined, config.confidence_level)?;

        let test_result = welch_t_test(baseline, treatment, config.alpha)?;
        let effect_size = cohens_d(baseline, treatment)?;

        // For latency benchmarks, lower is better
        let is_improvement = difference < 0.0;
        let is_significant = test_result.reject_null;

        Some(Self {
            baseline_stats,
            treatment_stats,
            difference,
            percent_change,
            confidence_interval: ci,
            test_result,
            effect_size,
            effect_interpretation: interpret_cohens_d(effect_size).to_string(),
            is_improvement,
            is_significant,
        })
    }
}

impl fmt::Display for BenchmarkComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Benchmark Comparison:")?;
        writeln!(f, "  Baseline:  {:.4} ± {:.4}", self.baseline_stats.mean, self.baseline_stats.std_dev)?;
        writeln!(f, "  Treatment: {:.4} ± {:.4}", self.treatment_stats.mean, self.treatment_stats.std_dev)?;
        writeln!(f, "  Difference: {:.4} ({:+.2}%)", self.difference, self.percent_change)?;
        writeln!(f, "  95% CI:    {}", self.confidence_interval)?;
        writeln!(f, "  P-value:   {:.6}", self.test_result.p_value)?;
        writeln!(f, "  Effect:    {:.4} ({})", self.effect_size, self.effect_interpretation)?;
        writeln!(
            f,
            "  Result:    {}{}",
            if self.is_significant { "Statistically significant" } else { "Not significant" },
            if self.is_significant && self.is_improvement {
                " improvement"
            } else if self.is_significant {
                " regression"
            } else {
                ""
            }
        )
    }
}

/// Validate benchmark results against claims
pub struct BenchmarkValidator {
    config: StatisticalConfig,
}

impl BenchmarkValidator {
    pub fn new(config: StatisticalConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(StatisticalConfig::default())
    }

    /// Validate that treatment is faster than baseline by at least `min_improvement`%
    pub fn validate_improvement(
        &self,
        baseline: &[f64],
        treatment: &[f64],
        min_improvement: f64,
    ) -> ValidationResult {
        let comparison = match BenchmarkComparison::compare(baseline, treatment, &self.config) {
            Some(c) => c,
            None => {
                return ValidationResult {
                    passed: false,
                    confidence_level: self.config.confidence_level,
                    p_value: None,
                    effect_size: None,
                    description: "Insufficient data for comparison".to_string(),
                }
            }
        };

        let improvement = -comparison.percent_change; // Negative change = improvement
        let passed = comparison.is_significant && improvement >= min_improvement;

        ValidationResult {
            passed,
            confidence_level: self.config.confidence_level,
            p_value: Some(comparison.test_result.p_value),
            effect_size: Some(comparison.effect_size),
            description: format!(
                "Observed {:.2}% improvement (required {:.2}%), {}",
                improvement,
                min_improvement,
                if comparison.is_significant {
                    "statistically significant"
                } else {
                    "not statistically significant"
                }
            ),
        }
    }

    /// Validate that variance is within acceptable bounds
    pub fn validate_variance(
        &self,
        data: &[f64],
        max_cv: f64, // Maximum coefficient of variation
    ) -> ValidationResult {
        let stats = match DescriptiveStats::compute(data) {
            Some(s) => s,
            None => {
                return ValidationResult {
                    passed: false,
                    confidence_level: self.config.confidence_level,
                    p_value: None,
                    effect_size: None,
                    description: "Insufficient data".to_string(),
                }
            }
        };

        let cv = stats.std_dev / stats.mean;
        let passed = cv <= max_cv;

        ValidationResult {
            passed,
            confidence_level: self.config.confidence_level,
            p_value: None,
            effect_size: None,
            description: format!(
                "Coefficient of variation: {:.4} (max allowed: {:.4})",
                cv, max_cv
            ),
        }
    }

    /// Validate scaling behavior (e.g., O(log n) vs O(n))
    pub fn validate_scaling(
        &self,
        sizes: &[f64],
        times: &[f64],
        expected_complexity: Complexity,
    ) -> ValidationResult {
        if sizes.len() != times.len() || sizes.len() < 3 {
            return ValidationResult {
                passed: false,
                confidence_level: self.config.confidence_level,
                p_value: None,
                effect_size: None,
                description: "Insufficient or mismatched data".to_string(),
            };
        }

        // Transform data according to expected complexity
        let transformed: Vec<f64> = sizes
            .iter()
            .map(|&n| expected_complexity.transform(n))
            .collect();

        // Compute correlation with transformed sizes
        let r = pearson_correlation(&transformed, times).unwrap_or(0.0);
        let r_squared = r * r;

        // Good fit if R² > 0.9
        let passed = r_squared > 0.9;

        ValidationResult {
            passed,
            confidence_level: self.config.confidence_level,
            p_value: None,
            effect_size: Some(r_squared),
            description: format!(
                "R² = {:.4} for {} scaling (good fit requires R² > 0.9)",
                r_squared, expected_complexity
            ),
        }
    }
}

/// Complexity class for scaling analysis
#[derive(Debug, Clone, Copy)]
pub enum Complexity {
    Constant,    // O(1)
    Logarithmic, // O(log n)
    Linear,      // O(n)
    Linearithmic, // O(n log n)
    Quadratic,   // O(n²)
}

impl Complexity {
    fn transform(&self, n: f64) -> f64 {
        match self {
            Complexity::Constant => 1.0,
            Complexity::Logarithmic => n.ln(),
            Complexity::Linear => n,
            Complexity::Linearithmic => n * n.ln(),
            Complexity::Quadratic => n * n,
        }
    }
}

impl fmt::Display for Complexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Complexity::Constant => write!(f, "O(1)"),
            Complexity::Logarithmic => write!(f, "O(log n)"),
            Complexity::Linear => write!(f, "O(n)"),
            Complexity::Linearithmic => write!(f, "O(n log n)"),
            Complexity::Quadratic => write!(f, "O(n²)"),
        }
    }
}

/// Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return None;
    }

    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}

/// Detect outliers using IQR method
pub fn detect_outliers_iqr(data: &[f64]) -> Vec<usize> {
    let stats = match DescriptiveStats::compute(data) {
        Some(s) => s,
        None => return vec![],
    };

    let lower_fence = stats.q1 - 1.5 * stats.iqr;
    let upper_fence = stats.q3 + 1.5 * stats.iqr;

    data.iter()
        .enumerate()
        .filter(|(_, &x)| x < lower_fence || x > upper_fence)
        .map(|(i, _)| i)
        .collect()
}

/// Remove outliers and return cleaned data
pub fn remove_outliers(data: &[f64]) -> Vec<f64> {
    let outlier_indices: std::collections::HashSet<_> = detect_outliers_iqr(data).into_iter().collect();
    data.iter()
        .enumerate()
        .filter(|(i, _)| !outlier_indices.contains(i))
        .map(|(_, &x)| x)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = DescriptiveStats::compute(&data).unwrap();

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_confidence_interval() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let ci = ConfidenceInterval::for_mean(&data, 0.95).unwrap();

        assert!(ci.contains(ci.point_estimate));
        assert!(ci.lower < ci.point_estimate);
        assert!(ci.upper > ci.point_estimate);
    }

    #[test]
    fn test_welch_t_test() {
        // Two clearly different samples
        let sample1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];

        let result = welch_t_test(&sample1, &sample2, 0.05).unwrap();
        assert!(result.reject_null);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_cohens_d() {
        let sample1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let d = cohens_d(&sample1, &sample2).unwrap();
        // Effect should be moderate
        assert!(d.abs() > 0.0);
    }

    #[test]
    fn test_benchmark_comparison() {
        let baseline: Vec<f64> = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let treatment: Vec<f64> = vec![80.0, 82.0, 78.0, 81.0, 79.0];

        let config = StatisticalConfig::default();
        let comparison = BenchmarkComparison::compare(&baseline, &treatment, &config).unwrap();

        assert!(comparison.is_improvement);
        assert!(comparison.is_significant);
        assert!(comparison.percent_change < 0.0); // Treatment is faster
    }

    #[test]
    fn test_outlier_detection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is outlier
        let outliers = detect_outliers_iqr(&data);

        assert!(!outliers.is_empty());
        assert!(outliers.contains(&5));
    }

    #[test]
    fn test_scaling_validation() {
        let validator = BenchmarkValidator::with_default_config();

        // Linear scaling data
        let sizes: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let times: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // Perfect linear

        let result = validator.validate_scaling(&sizes, &times, Complexity::Linear);
        assert!(result.passed);

        // Same data should not fit logarithmic well
        let result = validator.validate_scaling(&sizes, &times, Complexity::Logarithmic);
        // May or may not pass depending on fit, but effect_size gives R²
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_variance_validation() {
        let validator = BenchmarkValidator::with_default_config();

        // Low variance data
        let stable = vec![100.0, 101.0, 99.0, 100.5, 99.5];
        let result = validator.validate_variance(&stable, 0.1);
        assert!(result.passed);

        // High variance data
        let unstable = vec![50.0, 150.0, 30.0, 170.0, 100.0];
        let result = validator.validate_variance(&unstable, 0.1);
        assert!(!result.passed);
    }

    #[test]
    fn test_improvement_validation() {
        let validator = BenchmarkValidator::with_default_config();

        let baseline: Vec<f64> = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
        let treatment: Vec<f64> = vec![80.0, 82.0, 78.0, 81.0, 79.0, 80.0, 81.0, 79.0];

        let result = validator.validate_improvement(&baseline, &treatment, 15.0);
        assert!(result.passed); // ~20% improvement > 15% threshold
    }
}
