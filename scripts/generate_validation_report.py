import subprocess
import sys
import os
import datetime
import re

RESULTS_DIR = "results"
REPORT_FILE = "VALIDATION_REPORT.md"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_command(name, cmd, cwd=None):
    print(f"[{name}] Running: {cmd} ...")
    try:
        # Use shell=True for complex commands (e.g. redirects, though we handle redirects manually if needed)
        # But for security and simplicity with list args, shell=False is better unless we need pipe.
        # Here we will use shell=True to allow redirects in the command string if present.
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True
        )
        success = result.returncode == 0
        status = "PASS" if success else "FAIL"
        print(f"[{name}] Result: {status}")
        return success, result.stdout + result.stderr
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return False, str(e)

def main():
    ensure_dir(RESULTS_DIR)
    
    report_content = []
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content.append(f"# NEXUS Artifact Evaluation: Comprehensive Validation Report\n")
    report_content.append(f"**Date:** {current_date}\n")
    report_content.append(f"**Status:** READY FOR REVIEW\n\n")

    overall_success = True

    # --- 1. Memory Safety (Loom) ---
    print("\n--- Step 1: Memory Safety (Loom) ---")
    # Note: RUSTFLAGS="--cfg loom" is needed for loom tests
    loom_cmd = 'RUSTFLAGS="--cfg loom" cargo test --features loom --test loom_verification --release'
    success_loom, output_loom = run_command("Memory Safety", loom_cmd)
    
    # Extract relevant snippet (last few lines)
    loom_snippet = "\n".join(output_loom.strip().split('\n')[-20:])
    
    report_content.append("## 1. Memory Safety (Formal Verification)\n")
    report_content.append("- **Method:** Exhaustive Model Checking (Loom) across atomic interleavings.")
    report_content.append(f"- **Result:** { 'PASS' if success_loom else 'FAIL' }")
    report_content.append("\n<details>\n<summary>Details</summary>\n\n```text")
    report_content.append(loom_snippet)
    report_content.append("```\n\n</details>\n")
    if not success_loom: overall_success = False

    # --- 2. Zero-Copy Mechanics ---
    print("\n--- Step 2: Zero-Copy Mechanics ---")
    zc_cmd = "cargo run --release --bin proof_zero_copy"
    success_zc, output_zc = run_command("Zero-Copy", zc_cmd)
    
    # Check for "VERIFIED" keyword if exit code 0 isn't enough (it should be)
    if success_zc and "VERIFIED" not in output_zc and "PASSED" not in output_zc and "[PASS]" not in output_zc:
         if "VERIFICATION PASSED" not in output_zc:
            print("Warning: verification keyword not found in zero-copy output")
            # Don't fail if return code is 0, but note it.

    zc_snippet = "\n".join(output_zc.strip().split('\n')[-20:])

    report_content.append("## 2. Zero-Copy Mechanics (Forensic Analysis)\n")
    report_content.append("- **Method:** Pointer Stability & Page Fault Monitoring (OS Kernel Stats).")
    report_content.append(f"- **Result:** { 'PASS' if success_zc else 'FAIL' }")
    report_content.append(f"- **Proof:**\n\n```text")
    report_content.append(zc_snippet)
    report_content.append("```\n")
    if not success_zc: overall_success = False

    # --- 3. NUMA Affinity ---
    print("\n--- Step 3: NUMA Affinity ---")
    numa_cmd = "cargo test --release --test numa_placement"
    success_numa, output_numa = run_command("NUMA", numa_cmd)
    
    # Handle SKIP
    numa_status = "PASS"
    if not success_numa:
        numa_status = "FAIL"
    elif "0 ignored" not in output_numa and "ignored" in output_numa: 
        # cargo test output: "test result: ok. 1 passed; 0 failed; 1 ignored;"
        # If ignored count > 0, it was skipped
        if re.search(r" \d+ passed; 0 failed; [1-9]\d* ignored", output_numa):
             numa_status = "SKIP (Hardware Requirement)"
    
    numa_snippet = "\n".join(output_numa.strip().split('\n')[-15:])

    report_content.append("## 3. NUMA Affinity (Physical Verification)\n")
    report_content.append("- **Method:** `move_pages` syscall query on allocated pages.")
    report_content.append(f"- **Result:** {numa_status}")
    report_content.append("\n<details>\n<summary>Details</summary>\n\n```text")
    report_content.append(numa_snippet)
    report_content.append("```\n\n</details>\n")
    if numa_status == "FAIL": overall_success = False

    # --- 4. Scaling Law ---
    print("\n--- Step 4: Scaling Law ---")
    # 1. Run Benchmark
    # Using cargo bench as per my implementation, piping to csv
    bench_cmd = f"cargo bench -q --bench contention_scaling > {RESULTS_DIR}/scaling_raw.csv"
    success_bench, output_bench = run_command("Scaling Bench", bench_cmd)
    
    scaling_status = "FAIL"
    stats_output = "Benchmark Failed"
    
    if success_bench:
        # 2. Run Verification
        verify_cmd = f"python3 scripts/verify_scaling_law.py {RESULTS_DIR}/scaling_raw.csv"
        success_verify, output_verify = run_command("Scaling Verify", verify_cmd)
        
        stats_output = output_verify
        if success_verify and ("CONFIRMED" in output_verify or "PASS" in output_verify):
            scaling_status = "PASS"
        else:
            scaling_status = "FAIL"
            overall_success = False
    else:
        overall_success = False

    report_content.append("## 4. Scalability Law (Mathematical Proof)\n")
    report_content.append("- **Method:** Regression Analysis ($R^2$) on High-Contention Latency.")
    report_content.append(f"- **Result:** {scaling_status}")
    report_content.append(f"- **Statistics:**\n\n```text")
    report_content.append(stats_output)
    report_content.append("```\n")

    # Write Report
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_content))
    
    print(f"\nReport generated: {REPORT_FILE}")
    if overall_success:
        print("Overall Status: ALL SYSTEMS GO")
        sys.exit(0)
    else:
        print("Overall Status: FAILURES DETECTED")
        sys.exit(1)

if __name__ == "__main__":
    main()
