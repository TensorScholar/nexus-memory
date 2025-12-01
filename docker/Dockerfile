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

# Copy source files
COPY nexus-memory/ ./nexus-memory/
COPY nexus-benchmarks/ ./nexus-benchmarks/
COPY nexus-validation/ ./nexus-validation/
COPY baselines/ ./baselines/

# Build release binaries
RUN cd nexus-memory && cargo build --release
RUN cd nexus-benchmarks && cargo build --release 2>/dev/null || true

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
COPY --from=builder /workspace/nexus-memory/target/release/ ./nexus-memory/target/release/
COPY --from=builder /workspace/baselines/crossbeam-comparison ./baselines/
COPY --from=builder /workspace/baselines/hazard-pointer-baseline ./baselines/
COPY --from=builder /workspace/baselines/rcu-baseline ./baselines/

# Copy source and scripts
COPY nexus-memory/ ./nexus-memory/
COPY nexus-benchmarks/ ./nexus-benchmarks/
COPY nexus-validation/ ./nexus-validation/
COPY baselines/ ./baselines/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY README.md ./
COPY LICENSE ./

# Make scripts executable
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
LABEL org.opencontainers.image.source="https://github.com/nexus-memory/nexus"
