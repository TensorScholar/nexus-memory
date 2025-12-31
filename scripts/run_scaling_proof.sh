#!/bin/bash
set -e

echo "Starting O(log T) Scaling Proof..."
echo "1. Running contention benchmark (cargo bench)..."
# Using --bench contention_scaling. outputting CSV to stdout, saving to file
# Filters: -q (quiet compilation), --bench selection
# We pipe to a file first to treat it as input
cargo bench -q --bench contention_scaling > scaling_results.csv

echo "2. Benchmark complete. Verifying Scaling Law..."
# Run the verification script
python3 scripts/verify_scaling_law.py scaling_results.csv

# Cleanup
rm scaling_results.csv
echo "Done."
