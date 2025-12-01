#!/usr/bin/env python3
"""
Financial Risk Analytics Dataset Generator

Generates a realistic financial transaction dataset for benchmarking the NEXUS
memory reclamation framework. The generated data captures memory access patterns
(sparsity, cache locality) typical of real financial risk analytics workloads.

Dataset Schema:
- TransactionID: Unique identifier (UUID format)
- Amount: Transaction amount (follows heavy-tailed distribution)
- Timestamp: ISO format timestamp with millisecond precision
- CounterpartyID: Entity identifier (follows Zipfian distribution)
- InstrumentType: Financial instrument category
- RiskScore: Computed risk metric
- PortfolioID: Portfolio grouping for locality
- Region: Geographic region

Usage:
    python scripts/generate_financial_data.py [--output PATH] [--size SIZE_MB] [--seed SEED]

Example:
    python scripts/generate_financial_data.py --output data/financial_transactions.csv --size 1024

For paper reproduction:
    python scripts/generate_financial_data.py --size 1024 --seed 42
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("[ERROR] Required packages not found. Install with:")
    print("    pip install pandas numpy")
    sys.exit(1)


# Configuration constants
DEFAULT_OUTPUT = "nexus-benchmarks/data/financial_transactions.csv"
DEFAULT_SIZE_MB = 1024  # 1GB default
BYTES_PER_ROW_ESTIMATE = 120  # Approximate bytes per CSV row

# Distribution parameters for realistic financial data
AMOUNT_PARETO_ALPHA = 1.5  # Heavy-tailed transaction amounts
COUNTERPARTY_ZIPF_ALPHA = 1.2  # Power-law distribution for counterparties
NUM_COUNTERPARTIES = 10000
NUM_PORTFOLIOS = 500
NUM_INSTRUMENTS = 50
REGIONS = ["NA", "EU", "APAC", "LATAM", "MEA"]
INSTRUMENT_TYPES = [
    "EQUITY", "BOND", "DERIVATIVE", "FX", "COMMODITY", 
    "STRUCTURED", "SWAP", "OPTION", "FUTURE", "ETF"
]


def generate_transaction_ids(n: int, seed: int) -> np.ndarray:
    """Generate unique transaction IDs with temporal locality patterns."""
    rng = np.random.default_rng(seed)
    
    # Generate base IDs with some clustering (simulates batch processing)
    batch_size = 1000
    num_batches = (n + batch_size - 1) // batch_size
    
    ids = []
    for batch_idx in range(num_batches):
        batch_prefix = f"TXN{batch_idx:08d}"
        batch_count = min(batch_size, n - batch_idx * batch_size)
        for i in range(batch_count):
            # Add some randomness to simulate real-world ordering
            offset = rng.integers(0, 1000)
            ids.append(f"{batch_prefix}{i + offset:06d}")
    
    return np.array(ids[:n])


def generate_amounts(n: int, seed: int) -> np.ndarray:
    """Generate transaction amounts following a Pareto (heavy-tailed) distribution."""
    rng = np.random.default_rng(seed)
    
    # Pareto distribution for realistic financial transaction sizes
    # Most transactions are small, but some are very large
    base_amounts = (rng.pareto(AMOUNT_PARETO_ALPHA, n) + 1) * 1000
    
    # Add some noise and round to cents
    noise = rng.normal(1.0, 0.05, n)
    amounts = np.round(base_amounts * noise, 2)
    
    # Clip extreme values
    amounts = np.clip(amounts, 0.01, 1e9)
    
    return amounts


def generate_timestamps(n: int, seed: int, start_date: datetime) -> np.ndarray:
    """Generate timestamps with realistic temporal patterns (business hours, clusters)."""
    rng = np.random.default_rng(seed)
    
    timestamps = []
    current_time = start_date
    
    for i in range(n):
        # Simulate business hour clustering
        hour = rng.choice([9, 10, 11, 14, 15, 16], p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        minute = rng.integers(0, 60)
        second = rng.integers(0, 60)
        millisecond = rng.integers(0, 1000)
        
        # Move to next day periodically (simulate multi-day dataset)
        if i % 10000 == 0 and i > 0:
            current_time += timedelta(days=1)
            # Skip weekends
            while current_time.weekday() >= 5:
                current_time += timedelta(days=1)
        
        ts = current_time.replace(
            hour=hour, minute=minute, second=second, microsecond=millisecond * 1000
        )
        timestamps.append(ts.isoformat(timespec='milliseconds'))
    
    return np.array(timestamps)


def generate_counterparty_ids(n: int, seed: int) -> np.ndarray:
    """Generate counterparty IDs following Zipfian distribution (locality pattern)."""
    rng = np.random.default_rng(seed)
    
    # Zipf distribution: some counterparties appear much more frequently
    # This creates cache-friendly access patterns in real workloads
    zipf_samples = rng.zipf(COUNTERPARTY_ZIPF_ALPHA, n)
    counterparty_indices = np.minimum(zipf_samples - 1, NUM_COUNTERPARTIES - 1)
    
    counterparty_ids = np.array([f"CP{idx:06d}" for idx in counterparty_indices])
    
    return counterparty_ids


def generate_portfolio_ids(n: int, seed: int) -> np.ndarray:
    """Generate portfolio IDs with locality clustering."""
    rng = np.random.default_rng(seed)
    
    # Portfolios are accessed in clusters (locality pattern)
    cluster_size = 500
    num_clusters = (n + cluster_size - 1) // cluster_size
    
    portfolio_ids = []
    for cluster in range(num_clusters):
        # Each cluster focuses on a few portfolios
        dominant_portfolios = rng.choice(NUM_PORTFOLIOS, size=5, replace=False)
        cluster_count = min(cluster_size, n - cluster * cluster_size)
        
        # 80% from dominant portfolios, 20% random (cache-friendly pattern)
        for _ in range(cluster_count):
            if rng.random() < 0.8:
                portfolio_id = rng.choice(dominant_portfolios)
            else:
                portfolio_id = rng.integers(0, NUM_PORTFOLIOS)
            portfolio_ids.append(f"PF{portfolio_id:04d}")
    
    return np.array(portfolio_ids[:n])


def generate_instrument_types(n: int, seed: int) -> np.ndarray:
    """Generate instrument types with realistic distribution."""
    rng = np.random.default_rng(seed)
    
    # Weighted distribution: equities and bonds are more common
    weights = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    instrument_indices = rng.choice(len(INSTRUMENT_TYPES), size=n, p=weights)
    
    return np.array([INSTRUMENT_TYPES[i] for i in instrument_indices])


def generate_regions(n: int, seed: int) -> np.ndarray:
    """Generate geographic regions with realistic distribution."""
    rng = np.random.default_rng(seed)
    
    # Weighted by typical trading volume
    weights = [0.35, 0.30, 0.20, 0.10, 0.05]
    region_indices = rng.choice(len(REGIONS), size=n, p=weights)
    
    return np.array([REGIONS[i] for i in region_indices])


def generate_risk_scores(amounts: np.ndarray, seed: int) -> np.ndarray:
    """Generate risk scores based on amount and random factors."""
    rng = np.random.default_rng(seed)
    
    # Base risk from amount (log scale)
    base_risk = np.log10(amounts + 1) / 10.0
    
    # Add randomness
    noise = rng.normal(0, 0.1, len(amounts))
    risk_scores = np.clip(base_risk + noise, 0.0, 1.0)
    
    return np.round(risk_scores, 4)


def generate_dataset(
    target_size_mb: int, 
    seed: int = 42,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Generate the complete financial transaction dataset."""
    
    if start_date is None:
        start_date = datetime(2024, 1, 2, 9, 0, 0)  # Start on a Monday
    
    # Estimate number of rows needed
    target_bytes = target_size_mb * 1024 * 1024
    estimated_rows = target_bytes // BYTES_PER_ROW_ESTIMATE
    
    print(f"[INFO] Generating {estimated_rows:,} rows ({target_size_mb} MB target)")
    print(f"[INFO] Using seed: {seed}")
    
    # Generate columns with progress indication
    print("[1/8] Generating transaction IDs...")
    transaction_ids = generate_transaction_ids(estimated_rows, seed)
    
    print("[2/8] Generating amounts...")
    amounts = generate_amounts(estimated_rows, seed + 1)
    
    print("[3/8] Generating timestamps...")
    timestamps = generate_timestamps(estimated_rows, seed + 2, start_date)
    
    print("[4/8] Generating counterparty IDs...")
    counterparty_ids = generate_counterparty_ids(estimated_rows, seed + 3)
    
    print("[5/8] Generating instrument types...")
    instrument_types = generate_instrument_types(estimated_rows, seed + 4)
    
    print("[6/8] Generating portfolio IDs...")
    portfolio_ids = generate_portfolio_ids(estimated_rows, seed + 5)
    
    print("[7/8] Generating regions...")
    regions = generate_regions(estimated_rows, seed + 6)
    
    print("[8/8] Computing risk scores...")
    risk_scores = generate_risk_scores(amounts, seed + 7)
    
    print("[INFO] Assembling DataFrame...")
    df = pd.DataFrame({
        "TransactionID": transaction_ids,
        "Amount": amounts,
        "Timestamp": timestamps,
        "CounterpartyID": counterparty_ids,
        "InstrumentType": instrument_types,
        "RiskScore": risk_scores,
        "PortfolioID": portfolio_ids,
        "Region": regions,
    })
    
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Print dataset statistics for validation."""
    print("\n=== Dataset Statistics ===")
    print(f"Total rows: {len(df):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"\nAmount distribution:")
    print(f"  Min: ${df['Amount'].min():,.2f}")
    print(f"  Max: ${df['Amount'].max():,.2f}")
    print(f"  Mean: ${df['Amount'].mean():,.2f}")
    print(f"  Median: ${df['Amount'].median():,.2f}")
    print(f"\nUnique counterparties: {df['CounterpartyID'].nunique():,}")
    print(f"Unique portfolios: {df['PortfolioID'].nunique()}")
    print(f"\nInstrument type distribution:")
    print(df['InstrumentType'].value_counts().head(5))
    print(f"\nRegion distribution:")
    print(df['Region'].value_counts())
    print(f"\nRisk score distribution:")
    print(f"  Min: {df['RiskScore'].min():.4f}")
    print(f"  Max: {df['RiskScore'].max():.4f}")
    print(f"  Mean: {df['RiskScore'].mean():.4f}")
    print("===========================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic financial transaction dataset for NEXUS benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=DEFAULT_SIZE_MB,
        help=f"Target file size in MB (default: {DEFAULT_SIZE_MB})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print dataset statistics after generation"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("NEXUS Financial Transaction Dataset Generator")
    print("=" * 60)
    print(f"Target size: {args.size} MB")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Generate dataset
    df = generate_dataset(args.size, args.seed)
    
    # Validate if requested
    if args.validate:
        validate_dataset(df)
    
    # Save to CSV
    print(f"[INFO] Writing to {args.output}...")
    df.to_csv(output_path, index=False)
    
    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Generated {actual_size_mb:.2f} MB dataset with {len(df):,} rows")
    print(f"[INFO] File saved: {output_path.absolute()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
