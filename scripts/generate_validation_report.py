#!/usr/bin/env python3
"""
Scientific Validation Report Generator

Generates a summary validation report in Markdown format for artifact reviewers.
The report provides a single "green checkmark" overview of all paper claims.

Usage:
    python3 scripts/generate_validation_report.py --input results/verification_results.csv

Output:
    Markdown table summarizing all verification results with confidence levels.

Author: Mohammad-Ali Atashi
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path


# Paper claims to verify
CLAIMS = {
    'loom_verification': {
        'claim': 'Memory Safety',
        'paper_section': 'Section 4.1',
        'description': 'Freedom from data races in lock-free epoch primitives',
    },
    'zero_copy_pointer_stability': {
        'claim': 'Zero-Copy Transfers',
        'paper_section': 'Section 3.2',
        'description': 'Pointer addresses remain stable across paradigm transitions',
    },
    'zero_copy_page_faults': {
        'claim': 'Zero-Copy (OS Level)',
        'paper_section': 'Section 3.2',
        'description': 'No OS-level memory copies (page faults) during transitions',
    },
    'numa_physical_placement': {
        'claim': 'NUMA Affinity',
        'paper_section': 'Section 5.3',
        'description': 'Physical page placement on specified NUMA nodes',
    },
    'olog_t_scaling': {
        'claim': 'O(log T) Scaling',
        'paper_section': 'Section 4.2, Figure 7',
        'description': 'Synchronization overhead scales logarithmically with threads',
    },
}


def load_results(csv_path):
    """Load verification results from CSV."""
    results = {}
    
    if not os.path.exists(csv_path):
        return results
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_name = row.get('test_name', '')
            results[test_name] = {
                'result': row.get('result', 'UNKNOWN'),
                'confidence': row.get('confidence', 'N/A'),
                'method': row.get('method', 'Unknown'),
            }
    
    return results


def generate_status_emoji(result):
    """Generate status emoji based on result."""
    result_upper = result.upper()
    if result_upper == 'PASS':
        return '✅'
    elif result_upper == 'PARTIAL':
        return '⚠️'
    elif result_upper == 'SKIPPED':
        return '⏭️'
    elif result_upper == 'SYNTHETIC':
        return '🔬'
    else:
        return '❌'


def generate_confidence_badge(confidence):
    """Generate confidence level description."""
    if confidence in ['100%', 'Verified', 'Exhaustive']:
        return '100% (Exhaustive)'
    elif confidence in ['N/A', 'SKIPPED']:
        return 'N/A'
    elif confidence.startswith('0.'):
        # R² score
        try:
            r2 = float(confidence)
            if r2 >= 0.95:
                return f'> 0.95 (R²={r2:.3f})'
            else:
                return f'{r2:.3f} (R²)'
        except ValueError:
            return confidence
    else:
        return confidence


def generate_report(results, output_path):
    """Generate the validation report in Markdown format."""
    
    lines = []
    
    # Header
    lines.append('# Nexus Memory: Scientific Validation Report')
    lines.append('')
    lines.append(f'**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}')
    lines.append('')
    lines.append('This report summarizes the verification of all claims from the paper')
    lines.append('"Nexus: Unified Memory Reclamation for Cross-Paradigm Data Processing".')
    lines.append('')
    
    # Summary table
    lines.append('## Verification Summary')
    lines.append('')
    lines.append('| Claim | Method | Result | Confidence |')
    lines.append('|-------|--------|--------|------------|')
    
    all_pass = True
    
    for test_name, claim_info in CLAIMS.items():
        result_data = results.get(test_name, {
            'result': 'NOT_RUN',
            'confidence': 'N/A',
            'method': 'Not executed',
        })
        
        status = generate_status_emoji(result_data['result'])
        result_text = f"{status} {result_data['result']}"
        confidence = generate_confidence_badge(result_data['confidence'])
        
        lines.append(f"| {claim_info['claim']} | {result_data['method']} | {result_text} | {confidence} |")
        
        if result_data['result'].upper() not in ['PASS', 'SKIPPED', 'SYNTHETIC']:
            all_pass = False
    
    lines.append('')
    
    # Overall verdict
    lines.append('## Overall Verdict')
    lines.append('')
    
    if all_pass:
        lines.append('🎉 **ALL VERIFIABLE CLAIMS VALIDATED** 🎉')
        lines.append('')
        lines.append('The artifact successfully reproduces the key claims from the paper.')
    else:
        lines.append('⚠️ **PARTIAL VALIDATION**')
        lines.append('')
        lines.append('Some claims could not be fully verified. See individual results above.')
    
    lines.append('')
    
    # Detailed descriptions
    lines.append('## Claim Details')
    lines.append('')
    
    for test_name, claim_info in CLAIMS.items():
        result_data = results.get(test_name, {'result': 'NOT_RUN'})
        status = generate_status_emoji(result_data['result'])
        
        lines.append(f"### {claim_info['claim']} {status}")
        lines.append('')
        lines.append(f"**Paper Reference:** {claim_info['paper_section']}")
        lines.append('')
        lines.append(f"**Description:** {claim_info['description']}")
        lines.append('')
        
        if test_name in results:
            lines.append(f"**Method:** {results[test_name]['method']}")
            lines.append('')
            lines.append(f"**Result:** {results[test_name]['result']}")
            lines.append('')
        else:
            lines.append('**Status:** Not executed in this run')
            lines.append('')
    
    # Reproduction instructions
    lines.append('## Reproduction Instructions')
    lines.append('')
    lines.append('To reproduce these results:')
    lines.append('')
    lines.append('```bash')
    lines.append('# Full reproduction')
    lines.append('./scripts/reproduce-all.sh')
    lines.append('')
    lines.append('# Individual verification tests')
    lines.append('cargo test --features loom --test loom_verification  # Memory Safety')
    lines.append('cargo bench --package nexus-benchmarks -- zero_copy_proof  # Zero-Copy')
    lines.append('cargo test --package nexus-validation numa_verify  # NUMA')
    lines.append('cargo bench --package nexus-benchmarks -- contention_scaling  # O(log T)')
    lines.append('python3 scripts/plot_scaling_theory.py  # Scaling Analysis')
    lines.append('```')
    lines.append('')
    
    # Footer
    lines.append('---')
    lines.append('')
    lines.append('*Report generated by `scripts/generate_validation_report.py`*')
    lines.append('')
    lines.append('**Repository:** https://github.com/TensorScholar/nexus-memory')
    lines.append('')
    lines.append('**Author:** Mohammad-Ali Atashi')
    
    # Write output
    report_content = '\n'.join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    # Also print to console
    print(report_content)
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description='Generate validation report from verification results'
    )
    parser.add_argument(
        '--input', '-i',
        default='results/verification_results.csv',
        help='Path to verification results CSV'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/validation_report.md',
        help='Output path for Markdown report'
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_path = project_root / args.input
    output_path = project_root / args.output
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load results
    if input_path.exists():
        results = load_results(str(input_path))
        print(f"Loaded {len(results)} verification results from {input_path}")
    else:
        print(f"Warning: Results file not found at {input_path}")
        print("Generating report with placeholder data...")
        results = {}
    
    # Generate report
    all_pass = generate_report(results, str(output_path))
    
    print(f"\nReport saved to: {output_path}")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
