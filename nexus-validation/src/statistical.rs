//! Advanced statistical tests for benchmark validation
//!
//! Provides non-parametric tests, normality checks, and distribution analysis.

use crate::{DescriptiveStats, HypothesisTestResult};

/// Mann-Whitney U test (non-parametric alternative to t-test)
/// 
/// Tests whether two independent samples come from the same distribution.
pub fn mann_whitney_u(sample1: &[f64], sample2: &[f64], alpha: f64) -> Option<HypothesisTestResult> {
    let n1 = sample1.len();
    let n2 = sample2.len();

    if n1 < 2 || n2 < 2 {
        return None;
    }

    // Combine and rank
    let mut combined: Vec<(f64, usize)> = sample1
        .iter()
        .map(|&x| (x, 0))
        .chain(sample2.iter().map(|&x| (x, 1)))
        .collect();

    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (handling ties by averaging)
    let mut ranks = vec![0.0; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        while j < combined.len() && combined[j].0 == combined[i].0 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + 1..=j).map(|r| r as f64).sum::<f64>() / (j - i) as f64;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Sum of ranks for sample 1
    let r1: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|((_, group), _)| *group == 0)
        .map(|(_, &rank)| rank)
        .sum();

    // U statistics
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation for large samples
    let mean_u = (n1 * n2) as f64 / 2.0;
    let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();

    if std_u == 0.0 {
        return None;
    }

    let z = (u - mean_u) / std_u;
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

    Some(HypothesisTestResult {
        statistic: u,
        p_value,
        df: None,
        reject_null: p_value < alpha,
        alpha,
        test_name: "Mann-Whitney U Test".to_string(),
    })
}

/// Shapiro-Wilk test for normality (simplified approximation)
/// 
/// Tests whether a sample comes from a normally distributed population.
pub fn shapiro_wilk(data: &[f64], alpha: f64) -> Option<HypothesisTestResult> {
    let n = data.len();
    if n < 3 || n > 5000 {
        return None;
    }

    let _stats = DescriptiveStats::compute(data)?;

    // Sorted data
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Simplified W statistic using correlation with normal order statistics
    // This is an approximation of the full Shapiro-Wilk test
    let expected_normals: Vec<f64> = (1..=n)
        .map(|i| {
            let p = (i as f64 - 0.375) / (n as f64 + 0.25);
            inv_normal_cdf(p)
        })
        .collect();

    // Correlation between sorted data and expected normal order statistics
    let mean_data: f64 = sorted.iter().sum::<f64>() / n as f64;
    let mean_exp: f64 = expected_normals.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_data = 0.0;
    let mut var_exp = 0.0;

    for i in 0..n {
        let d_data = sorted[i] - mean_data;
        let d_exp = expected_normals[i] - mean_exp;
        cov += d_data * d_exp;
        var_data += d_data * d_data;
        var_exp += d_exp * d_exp;
    }

    if var_data == 0.0 || var_exp == 0.0 {
        return None;
    }

    let r = cov / (var_data.sqrt() * var_exp.sqrt());
    let w = r * r;

    // P-value approximation (using normal transformation)
    // This is a rough approximation for demonstration
    let ln_w = (1.0 - w).ln();
    let mu = 0.0;
    let sigma = 1.0;
    let z = (ln_w - mu) / sigma;
    let p_value = normal_cdf(z);

    Some(HypothesisTestResult {
        statistic: w,
        p_value,
        df: Some(n as f64),
        reject_null: p_value < alpha,
        alpha,
        test_name: "Shapiro-Wilk Normality Test".to_string(),
    })
}

/// Levene's test for homogeneity of variances
pub fn levene_test(samples: &[&[f64]], alpha: f64) -> Option<HypothesisTestResult> {
    let k = samples.len();
    if k < 2 {
        return None;
    }

    // Compute group medians
    let medians: Vec<f64> = samples
        .iter()
        .filter_map(|s| DescriptiveStats::compute(s).map(|stats| stats.median))
        .collect();

    if medians.len() != k {
        return None;
    }

    // Compute deviations from medians
    let deviations: Vec<Vec<f64>> = samples
        .iter()
        .zip(medians.iter())
        .map(|(sample, &median)| sample.iter().map(|&x| (x - median).abs()).collect())
        .collect();

    // Compute group means of deviations
    let group_means: Vec<f64> = deviations
        .iter()
        .map(|d| d.iter().sum::<f64>() / d.len() as f64)
        .collect();

    // Grand mean of deviations
    let n_total: usize = samples.iter().map(|s| s.len()).sum();
    let grand_mean: f64 = deviations.iter().flatten().sum::<f64>() / n_total as f64;

    // Between-group sum of squares
    let ss_between: f64 = samples
        .iter()
        .zip(group_means.iter())
        .map(|(s, &mean)| s.len() as f64 * (mean - grand_mean).powi(2))
        .sum();

    // Within-group sum of squares
    let ss_within: f64 = deviations
        .iter()
        .zip(group_means.iter())
        .map(|(d, &mean)| d.iter().map(|&x| (x - mean).powi(2)).sum::<f64>())
        .sum();

    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;

    if df_within <= 0.0 || ss_within == 0.0 {
        return None;
    }

    let f_stat = (ss_between / df_between) / (ss_within / df_within);

    // P-value from F-distribution (using approximation)
    let p_value = 1.0 - f_cdf(f_stat, df_between, df_within);

    Some(HypothesisTestResult {
        statistic: f_stat,
        p_value,
        df: Some(df_between),
        reject_null: p_value < alpha,
        alpha,
        test_name: "Levene's Test for Equality of Variances".to_string(),
    })
}

/// Kruskal-Wallis H test (non-parametric one-way ANOVA)
pub fn kruskal_wallis(samples: &[&[f64]], alpha: f64) -> Option<HypothesisTestResult> {
    let k = samples.len();
    if k < 2 {
        return None;
    }

    // Combine all samples with group labels
    let mut combined: Vec<(f64, usize)> = samples
        .iter()
        .enumerate()
        .flat_map(|(group, sample)| sample.iter().map(move |&x| (x, group)))
        .collect();

    let n = combined.len();
    if n < k + 1 {
        return None;
    }

    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (handling ties)
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let avg_rank = (i + 1..=j).map(|r| r as f64).sum::<f64>() / (j - i) as f64;
        for idx in i..j {
            ranks[idx] = avg_rank;
        }
        i = j;
    }

    // Sum of ranks per group
    let mut rank_sums = vec![0.0; k];
    let group_sizes: Vec<usize> = samples.iter().map(|s| s.len()).collect();

    for (idx, &(_, group)) in combined.iter().enumerate() {
        rank_sums[group] += ranks[idx];
    }

    // H statistic
    let n_f = n as f64;
    let sum_term: f64 = rank_sums
        .iter()
        .zip(group_sizes.iter())
        .map(|(&r, &ni)| r * r / ni as f64)
        .sum();

    let h = (12.0 / (n_f * (n_f + 1.0))) * sum_term - 3.0 * (n_f + 1.0);

    // P-value from chi-squared distribution
    let df = (k - 1) as f64;
    let p_value = 1.0 - chi_squared_cdf(h, df);

    Some(HypothesisTestResult {
        statistic: h,
        p_value,
        df: Some(df),
        reject_null: p_value < alpha,
        alpha,
        test_name: "Kruskal-Wallis H Test".to_string(),
    })
}

/// Effect size: Cliff's delta (non-parametric)
/// 
/// Measures the probability that a randomly selected value from one group
/// is greater than a randomly selected value from another group.
pub fn cliffs_delta(sample1: &[f64], sample2: &[f64]) -> Option<f64> {
    if sample1.is_empty() || sample2.is_empty() {
        return None;
    }

    let mut greater = 0;
    let mut less = 0;

    for &x in sample1 {
        for &y in sample2 {
            if x > y {
                greater += 1;
            } else if x < y {
                less += 1;
            }
        }
    }

    let n = (sample1.len() * sample2.len()) as f64;
    Some((greater as f64 - less as f64) / n)
}

/// Interpret Cliff's delta effect size
pub fn interpret_cliffs_delta(d: f64) -> &'static str {
    let d = d.abs();
    if d < 0.147 {
        "negligible"
    } else if d < 0.33 {
        "small"
    } else if d < 0.474 {
        "medium"
    } else {
        "large"
    }
}

/// Hodges-Lehmann estimator (robust location estimator)
pub fn hodges_lehmann(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let n = data.len();
    let mut pairwise_means = Vec::with_capacity(n * (n + 1) / 2);

    for i in 0..n {
        for j in i..n {
            pairwise_means.push((data[i] + data[j]) / 2.0);
        }
    }

    pairwise_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = pairwise_means.len() / 2;
    if pairwise_means.len() % 2 == 0 {
        Some((pairwise_means[mid - 1] + pairwise_means[mid]) / 2.0)
    } else {
        Some(pairwise_means[mid])
    }
}

/// Median Absolute Deviation (MAD) - robust measure of scale
pub fn mad(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let stats = DescriptiveStats::compute(data)?;
    let mut deviations: Vec<f64> = data.iter().map(|&x| (x - stats.median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = deviations.len();
    if n % 2 == 0 {
        Some((deviations[n / 2 - 1] + deviations[n / 2]) / 2.0)
    } else {
        Some(deviations[n / 2])
    }
}

/// Robust coefficient of variation using MAD
pub fn robust_cv(data: &[f64]) -> Option<f64> {
    let stats = DescriptiveStats::compute(data)?;
    let mad_value = mad(data)?;

    if stats.median == 0.0 {
        return None;
    }

    // Scale factor to make MAD comparable to standard deviation
    // For normal distribution, MAD ≈ 0.6745 * σ
    Some((mad_value * 1.4826) / stats.median)
}

// Helper functions

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

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

fn erf(x: f64) -> f64 {
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

/// Chi-squared CDF approximation
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Use normal approximation for large df
    if df > 100.0 {
        let z = (x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df));
        let se = (2.0 / (9.0 * df)).sqrt();
        return normal_cdf(z / se);
    }
    // Simple approximation using incomplete gamma function
    lower_incomplete_gamma(df / 2.0, x / 2.0) / gamma(df / 2.0)
}

/// F-distribution CDF approximation
fn f_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = df1 / 2.0;
    let b = df2 / 2.0;
    let z = df1 * x / (df1 * x + df2);
    regularized_incomplete_beta(a, b, z)
}

/// Lower incomplete gamma function (approximation)
fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    // Series expansion
    let mut sum = 0.0;
    let mut term = 1.0 / a;
    sum += term;
    for n in 1..100 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < 1e-10 {
            break;
        }
    }
    sum * (-x).exp() * x.powf(a)
}

/// Gamma function (Stirling approximation for larger values)
fn gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
    }
    // Stirling approximation
    let x = x - 1.0;
    let g = 7;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut y = c[0];
    for i in 1..g + 2 {
        y += c[i] / (x + i as f64);
    }

    let t = x + g as f64 + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * y
}

/// Regularized incomplete beta function (approximation)
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Continued fraction approximation
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

fn ln_gamma(x: f64) -> f64 {
    gamma(x).ln()
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < eps {
        d = eps;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;

        // Even step
        let an = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + an * d;
        if d.abs() < eps {
            d = eps;
        }
        c = 1.0 + an / c;
        if c.abs() < eps {
            c = eps;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let an = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + an * d;
        if d.abs() < eps {
            d = eps;
        }
        c = 1.0 + an / c;
        if c.abs() < eps {
            c = eps;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mann_whitney() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result = mann_whitney_u(&sample1, &sample2, 0.05).unwrap();
        assert!(result.reject_null);
    }

    #[test]
    fn test_kruskal_wallis() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];
        let group3 = vec![7.0, 8.0, 9.0];

        let samples: Vec<&[f64]> = vec![&group1, &group2, &group3];
        let result = kruskal_wallis(&samples, 0.05).unwrap();

        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_cliffs_delta() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![4.0, 5.0, 6.0];

        let delta = cliffs_delta(&sample1, &sample2).unwrap();
        assert!(delta < 0.0); // sample1 values < sample2 values
        assert_eq!(interpret_cliffs_delta(delta), "large");
    }

    #[test]
    fn test_hodges_lehmann() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hl = hodges_lehmann(&data).unwrap();
        assert!((hl - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_mad() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mad_value = mad(&data).unwrap();
        assert!(mad_value > 0.0);
    }

    #[test]
    fn test_robust_cv() {
        let data = vec![100.0, 101.0, 99.0, 100.5, 99.5];
        let cv = robust_cv(&data).unwrap();
        assert!(cv < 0.1); // Low variance
    }
}
