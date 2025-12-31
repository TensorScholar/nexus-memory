#!/usr/bin/env python3
"""
O(log T) Scaling Theory Validation

This script analyzes the contention scaling benchmark results to statistically
prove that the epoch synchronization overhead follows O(log T) scaling.

Methodology:
1. Load benchmark results from CSV
2. Fit logarithmic curve: y = a * log(x) + b
3. Calculate R² score to validate the fit quality
4. Compare with linear fit to demonstrate superiority of log model

Output:
- Console summary with statistical analysis
- PDF plot showing data points and fitted curve
- Confidence in the O(log T) claim based on R² score

Usage:
    python3 scripts/plot_scaling_theory.py

Requirements:
    pip install pandas numpy scipy matplotlib

Author: Mohammad-Ali Atashi
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use('Agg')


def logarithmic_model(x, a, b):
    """
    Logarithmic model: y = a * log(x) + b
    
    This represents O(log T) scaling behavior.
    """
    return a * np.log(x) + b


def linear_model(x, a, b):
    """
    Linear model: y = a * x + b
    
    This represents O(T) scaling behavior (baseline comparison).
    """
    return a * x + b


def fit_logarithmic(x, y):
    """
    Fit a logarithmic curve to the data.
    
    Returns:
        params: (a, b) coefficients
        r_squared: R² score
        residuals: Fitting residuals
    """
    try:
        params, covariance = optimize.curve_fit(
            logarithmic_model, 
            x, 
            y, 
            p0=[1.0, 0.0],
            maxfev=10000
        )
        
        # Calculate R²
        y_pred = logarithmic_model(x, *params)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        residuals = y - y_pred
        
        return params, r_squared, residuals
    except Exception as e:
        print(f"Warning: Logarithmic fit failed: {e}")
        return (0, 0), 0, np.zeros_like(y)


def fit_linear(x, y):
    """
    Fit a linear curve to the data.
    
    Returns:
        params: (a, b) coefficients
        r_squared: R² score
    """
    try:
        params, covariance = optimize.curve_fit(
            linear_model, 
            x, 
            y, 
            p0=[1.0, 0.0],
            maxfev=10000
        )
        
        # Calculate R²
        y_pred = linear_model(x, *params)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return params, r_squared
    except Exception as e:
        print(f"Warning: Linear fit failed: {e}")
        return (0, 0), 0


def analyze_scaling(df, output_dir):
    """
    Perform comprehensive scaling analysis.
    
    Args:
        df: DataFrame with columns 'threads' and latency metrics
        output_dir: Directory for output files
    """
    # Extract data
    threads = df['threads'].values.astype(float)
    
    # Use combined latency if available, otherwise pin latency
    if 'avg_pin_latency_ns' in df.columns:
        pin_latency = df['avg_pin_latency_ns'].values
    else:
        pin_latency = np.zeros(len(threads))
    
    if 'avg_advance_latency_ns' in df.columns:
        advance_latency = df['avg_advance_latency_ns'].values
    else:
        advance_latency = np.zeros(len(threads))
    
    # Combined latency for analysis
    combined_latency = pin_latency + advance_latency
    
    print("\n" + "=" * 60)
    print("O(log T) SCALING VALIDATION REPORT")
    print("=" * 60)
    
    # Analyze pin() latency
    print("\n--- Pin Latency Analysis ---")
    log_params, log_r2, _ = fit_logarithmic(threads, pin_latency)
    lin_params, lin_r2 = fit_linear(threads, pin_latency)
    
    print(f"Logarithmic fit: y = {log_params[0]:.4f} * log(x) + {log_params[1]:.4f}")
    print(f"  R² = {log_r2:.6f}")
    print(f"Linear fit: y = {lin_params[0]:.4f} * x + {lin_params[1]:.4f}")
    print(f"  R² = {lin_r2:.6f}")
    print(f"Log model {'BETTER' if log_r2 > lin_r2 else 'worse'} than linear")
    
    # Analyze try_advance() latency
    print("\n--- Advance Latency Analysis ---")
    log_params_adv, log_r2_adv, _ = fit_logarithmic(threads, advance_latency)
    lin_params_adv, lin_r2_adv = fit_linear(threads, advance_latency)
    
    print(f"Logarithmic fit: y = {log_params_adv[0]:.4f} * log(x) + {log_params_adv[1]:.4f}")
    print(f"  R² = {log_r2_adv:.6f}")
    print(f"Linear fit: y = {lin_params_adv[0]:.4f} * x + {lin_params_adv[1]:.4f}")
    print(f"  R² = {lin_r2_adv:.6f}")
    
    # Analyze combined latency
    print("\n--- Combined Latency Analysis ---")
    log_params_comb, log_r2_comb, residuals = fit_logarithmic(threads, combined_latency)
    lin_params_comb, lin_r2_comb = fit_linear(threads, combined_latency)
    
    print(f"Logarithmic fit: y = {log_params_comb[0]:.4f} * log(x) + {log_params_comb[1]:.4f}")
    print(f"  R² = {log_r2_comb:.6f}")
    print(f"Linear fit: y = {lin_params_comb[0]:.4f} * x + {lin_params_comb[1]:.4f}")
    print(f"  R² = {lin_r2_comb:.6f}")
    
    # Statistical significance of logarithmic model
    r2_improvement = log_r2_comb - lin_r2_comb
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if log_r2_comb >= 0.95:
        conclusion = "STRONG EVIDENCE for O(log T) scaling"
        confidence = "HIGH (R² ≥ 0.95)"
    elif log_r2_comb >= 0.80:
        conclusion = "MODERATE EVIDENCE for O(log T) scaling"
        confidence = "MEDIUM (0.80 ≤ R² < 0.95)"
    else:
        conclusion = "WEAK EVIDENCE for O(log T) scaling"
        confidence = "LOW (R² < 0.80)"
    
    print(f"\n{conclusion}")
    print(f"Confidence: {confidence}")
    print(f"Combined R² (logarithmic): {log_r2_comb:.6f}")
    print(f"R² improvement over linear: {r2_improvement:+.6f}")
    
    # Generate plots
    generate_plots(
        threads, pin_latency, advance_latency, combined_latency,
        log_params, log_params_adv, log_params_comb,
        log_r2, log_r2_adv, log_r2_comb,
        output_dir
    )
    
    # Return summary for validation report
    return {
        'pin_r2': log_r2,
        'advance_r2': log_r2_adv,
        'combined_r2': log_r2_comb,
        'log_better_than_linear': log_r2_comb > lin_r2_comb,
        'conclusion': conclusion,
    }


def generate_plots(threads, pin_lat, adv_lat, comb_lat,
                   log_params, log_params_adv, log_params_comb,
                   r2_pin, r2_adv, r2_comb,
                   output_dir):
    """
    Generate publication-quality plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extended x range for fitted curves
    x_fit = np.linspace(1, max(threads) * 1.1, 100)
    
    # Pin latency plot
    ax1 = axes[0]
    ax1.scatter(threads, pin_lat, s=80, c='#2E86AB', marker='o', 
                label='Measured', zorder=3)
    ax1.plot(x_fit, logarithmic_model(x_fit, *log_params), 
             'r-', linewidth=2, label=f'Log fit (R²={r2_pin:.3f})')
    ax1.set_xlabel('Thread Count (T)')
    ax1.set_ylabel('Average Latency (ns)')
    ax1.set_title('Pin() Latency Scaling')
    ax1.legend()
    ax1.set_xscale('log', base=2)
    
    # Advance latency plot
    ax2 = axes[1]
    ax2.scatter(threads, adv_lat, s=80, c='#E94F37', marker='s',
                label='Measured', zorder=3)
    ax2.plot(x_fit, logarithmic_model(x_fit, *log_params_adv),
             'b-', linewidth=2, label=f'Log fit (R²={r2_adv:.3f})')
    ax2.set_xlabel('Thread Count (T)')
    ax2.set_ylabel('Average Latency (ns)')
    ax2.set_title('Try_Advance() Latency Scaling')
    ax2.legend()
    ax2.set_xscale('log', base=2)
    
    # Combined latency plot
    ax3 = axes[2]
    ax3.scatter(threads, comb_lat, s=80, c='#44AF69', marker='^',
                label='Measured', zorder=3)
    ax3.plot(x_fit, logarithmic_model(x_fit, *log_params_comb),
             'k-', linewidth=2, label=f'O(log T) fit (R²={r2_comb:.3f})')
    ax3.set_xlabel('Thread Count (T)')
    ax3.set_ylabel('Combined Latency (ns)')
    ax3.set_title('Total Synchronization Overhead')
    ax3.legend()
    ax3.set_xscale('log', base=2)
    
    # Add O(log T) annotation
    ax3.annotate(
        f'y = {log_params_comb[0]:.2f}·log(T) + {log_params_comb[1]:.2f}',
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save as PDF for publication quality
    pdf_path = os.path.join(output_dir, 'scaling_analysis.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {pdf_path}")
    
    # Also save as PNG for quick viewing
    png_path = os.path.join(output_dir, 'scaling_analysis.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Validate O(log T) scaling from benchmark results'
    )
    parser.add_argument(
        '--input', '-i',
        default='nexus-benchmarks/results/internal_contention.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='figures',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--synthetic', '-s',
        action='store_true',
        help='Generate synthetic data for testing'
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Determine input path
    if args.synthetic:
        print("Generating synthetic O(log T) data for demonstration...")
        # Create synthetic data that follows O(log T) + noise
        threads = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        np.random.seed(42)
        
        # O(log T) with realistic parameters and noise
        base_latency = 50.0
        log_coeff = 15.0
        noise_scale = 5.0
        
        pin_latency = (base_latency + log_coeff * np.log(threads) + 
                       np.random.normal(0, noise_scale, len(threads)))
        advance_latency = (base_latency * 1.5 + log_coeff * 1.2 * np.log(threads) +
                           np.random.normal(0, noise_scale * 1.5, len(threads)))
        
        df = pd.DataFrame({
            'threads': threads,
            'avg_pin_latency_ns': np.maximum(pin_latency, 10),  # Ensure positive
            'avg_advance_latency_ns': np.maximum(advance_latency, 10),
        })
    else:
        input_path = project_root / args.input
        
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            print("Run the contention scaling benchmark first:")
            print("  cargo bench --package nexus-benchmarks -- contention_scaling")
            print("\nOr use --synthetic to generate test data.")
            sys.exit(1)
        
        print(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
    
    print(f"Data points: {len(df)}")
    print(f"Thread counts: {df['threads'].tolist()}")
    
    output_dir = project_root / args.output_dir
    results = analyze_scaling(df, str(output_dir))
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    # Return success if R² > 0.80
    return 0 if results['combined_r2'] >= 0.80 else 1


if __name__ == '__main__':
    sys.exit(main())
