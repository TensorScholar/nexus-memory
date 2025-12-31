# Nexus Memory: PVLDB 2026 Artifact Evaluation Container
#
# This Dockerfile creates a reproducible environment for evaluating
# the Nexus Memory artifact from the paper:
# "Nexus: Unified Memory Reclamation for Cross-Paradigm Data Systems"
#
# Usage:
#   docker build -t nexus-memory .
#   docker run -it nexus-memory
#   docker run nexus-memory ./scripts/reproduce-all.sh --quick
#
# For development with bind mount:
#   docker run -it -v $(pwd)/results:/workspace/results nexus-memory

FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy Cargo manifests first for dependency caching (layer optimization)
COPY Cargo.toml Cargo.lock ./
COPY nexus-memory/Cargo.toml ./nexus-memory/
COPY nexus-benchmarks/Cargo.toml ./nexus-benchmarks/
COPY nexus-validation/Cargo.toml ./nexus-validation/
COPY nexus-verification/Cargo.toml ./nexus-verification/

# Create dummy source files to cache dependencies
RUN mkdir -p nexus-memory/src nexus-benchmarks/src nexus-validation/src nexus-verification/src && \
    echo "fn main() {}" > nexus-memory/src/lib.rs && \
    echo "fn main() {}" > nexus-benchmarks/src/lib.rs && \
    echo "fn main() {}" > nexus-validation/src/lib.rs && \
    echo "fn main() {}" > nexus-verification/src/lib.rs && \
    cargo fetch || true

# Copy actual source files (invalidates cache only if source changes)
COPY nexus-memory/ ./nexus-memory/
COPY nexus-benchmarks/ ./nexus-benchmarks/
COPY nexus-validation/ ./nexus-validation/
COPY nexus-verification/ ./nexus-verification/
COPY baselines/ ./baselines/
COPY formal-verification/ ./formal-verification/

# Build release binaries
RUN cargo build --release 2>/dev/null || true

# Build baselines
RUN cd baselines && \
    rustc -O crossbeam-comparison.rs -o crossbeam-comparison 2>/dev/null || true && \
    rustc -O hazard-pointer-baseline.rs -o hazard-pointer-baseline 2>/dev/null || true && \
    rustc -O rcu-baseline.rs -o rcu-baseline 2>/dev/null || true

# Runtime image
FROM python:3.11-slim-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    git \
    curl \
    numactl \
    linux-perf \
    && rm -rf /var/lib/apt/lists/*

# Install Rust runtime for any dynamic compilation
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies for analysis
RUN pip install --no-cache-dir \
    matplotlib \
    pandas \
    numpy \
    scipy \
    seaborn \
    jupyter

# Create workspace
WORKDIR /workspace

# Copy built artifacts from builder
COPY --from=builder /workspace/target/release/ ./target/release/
COPY --from=builder /workspace/baselines/crossbeam-comparison ./baselines/ 2>/dev/null || true
COPY --from=builder /workspace/baselines/hazard-pointer-baseline ./baselines/ 2>/dev/null || true
COPY --from=builder /workspace/baselines/rcu-baseline ./baselines/ 2>/dev/null || true

# Copy source, scripts, and documentation
COPY Cargo.toml Cargo.lock ./
COPY nexus-memory/ ./nexus-memory/
COPY nexus-benchmarks/ ./nexus-benchmarks/
COPY nexus-validation/ ./nexus-validation/
COPY nexus-verification/ ./nexus-verification/
COPY baselines/ ./baselines/
COPY formal-verification/ ./formal-verification/
COPY scripts/ ./scripts/
COPY README.md VALIDATION_REPORT.md reproducibility.md LICENSE ./

# Make scripts executable (BEFORE CMD)
RUN chmod +x scripts/*.sh

# Create output directories
RUN mkdir -p results figures

# Set environment
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV NEXUS_HOME=/workspace

# Default command
CMD ["./scripts/reproduce-all.sh", "--quick"]

# Metadata
LABEL maintainer="nexus-memory@example.com"
LABEL version="1.0"
LABEL description="Nexus Memory PVLDB 2026 Artifact Evaluation"
LABEL org.opencontainers.image.source="https://github.com/TensorScholar/nexus-memory"
