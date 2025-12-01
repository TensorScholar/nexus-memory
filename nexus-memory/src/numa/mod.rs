//! NUMA-Aware Memory Allocation
//!
//! This module provides NUMA (Non-Uniform Memory Access) aware memory allocation
//! to optimize data placement for multi-socket systems. The key insight is that
//! memory access latency varies significantly based on which NUMA node owns the
//! memory versus which node's CPU is accessing it.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐
//! │   NUMA Node 0   │     │   NUMA Node 1   │
//! │ ┌─────────────┐ │     │ ┌─────────────┐ │
//! │ │ CPU 0-7     │ │     │ │ CPU 8-15    │ │
//! │ └─────────────┘ │     │ └─────────────┘ │
//! │ ┌─────────────┐ │     │ ┌─────────────┐ │
//! │ │ Local RAM   │◄├─────┤►│ Local RAM   │ │
//! │ │ (Fast)      │ │ QPI │ │ (Fast)      │ │
//! │ └─────────────┘ │     │ └─────────────┘ │
//! └─────────────────┘     └─────────────────┘
//! ```
//!
//! # Performance Model
//!
//! Memory access latency follows:
//! ```text
//! L(src, dst) = L_base + α × d(src, dst) + β × congestion(src, dst)
//! ```
//!
//! Where `d(src, dst)` is the topological distance between nodes.
//!
//! # Platform Support
//!
//! - Linux: Full support via libnuma
//! - macOS: Simulated (single NUMA node)
//! - Windows: Partial support via Windows API

mod topology;
mod allocator;

pub use topology::{NumaTopology, NodeId, CpuSet};
pub use allocator::{NumaAllocator, AllocationPolicy};

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Maximum supported NUMA nodes
pub const MAX_NUMA_NODES: usize = 64;

/// Error types for NUMA operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumaError {
    /// NUMA is not available on this system
    Unavailable,
    
    /// Invalid NUMA node ID
    InvalidNode(u32),
    
    /// Memory allocation failed
    AllocationFailed,
    
    /// Thread affinity operation failed
    AffinityError,
    
    /// Topology discovery failed
    TopologyError,
    
    /// Operation not supported on this platform
    NotSupported,
}

impl core::fmt::Display for NumaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NumaError::Unavailable => write!(f, "NUMA not available"),
            NumaError::InvalidNode(n) => write!(f, "invalid NUMA node: {}", n),
            NumaError::AllocationFailed => write!(f, "NUMA allocation failed"),
            NumaError::AffinityError => write!(f, "thread affinity error"),
            NumaError::TopologyError => write!(f, "topology discovery failed"),
            NumaError::NotSupported => write!(f, "operation not supported"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NumaError {}

/// Result type for NUMA operations
pub type Result<T> = core::result::Result<T, NumaError>;

/// A NUMA node with its associated resources.
///
/// Represents a single NUMA domain with its local memory and CPUs.
#[derive(Debug)]
pub struct NumaNode {
    /// Node identifier
    id: u32,
    
    /// CPUs associated with this node
    cpus: CpuSet,
    
    /// Total memory in bytes
    total_memory: u64,
    
    /// Available memory in bytes
    available_memory: AtomicU64,
    
    /// Distance to other nodes (index = node id, value = distance)
    distances: [u8; MAX_NUMA_NODES],
}

impl NumaNode {
    /// Creates a new NUMA node.
    pub(crate) fn new(id: u32, cpus: CpuSet, total_memory: u64, distances: [u8; MAX_NUMA_NODES]) -> Self {
        Self {
            id,
            cpus,
            total_memory,
            available_memory: AtomicU64::new(total_memory),
            distances,
        }
    }

    /// Returns the node ID.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the CPUs associated with this node.
    #[inline]
    pub fn cpus(&self) -> &CpuSet {
        &self.cpus
    }

    /// Returns the total memory in bytes.
    #[inline]
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    /// Returns the available memory in bytes.
    #[inline]
    pub fn available_memory(&self) -> u64 {
        self.available_memory.load(Ordering::Relaxed)
    }

    /// Returns the distance to another node.
    #[inline]
    pub fn distance_to(&self, other: u32) -> u8 {
        if other as usize >= MAX_NUMA_NODES {
            u8::MAX
        } else {
            self.distances[other as usize]
        }
    }

    /// Returns whether this is a local access (same node).
    #[inline]
    pub fn is_local(&self, other: u32) -> bool {
        self.id == other
    }
}

/// Memory policy for NUMA allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPolicy {
    /// Allocate on the local node (default)
    Local,
    
    /// Bind to a specific node
    Bind(u32),
    
    /// Prefer a specific node but allow fallback
    Prefer(u32),
    
    /// Interleave pages across all nodes
    Interleave,
    
    /// Interleave across specified nodes
    InterleaveNodes(u64), // Bitmask of nodes
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self::Local
    }
}

/// Statistics for NUMA memory operations.
#[derive(Debug, Default)]
pub struct NumaStats {
    /// Total bytes allocated per node
    pub bytes_allocated: [AtomicU64; MAX_NUMA_NODES],
    
    /// Total allocations per node
    pub allocation_count: [AtomicU64; MAX_NUMA_NODES],
    
    /// Local memory accesses
    pub local_accesses: AtomicU64,
    
    /// Remote memory accesses
    pub remote_accesses: AtomicU64,
}

impl NumaStats {
    /// Records an allocation.
    pub fn record_allocation(&self, node: u32, bytes: u64) {
        if (node as usize) < MAX_NUMA_NODES {
            self.bytes_allocated[node as usize].fetch_add(bytes, Ordering::Relaxed);
            self.allocation_count[node as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a deallocation.
    pub fn record_deallocation(&self, node: u32, bytes: u64) {
        if (node as usize) < MAX_NUMA_NODES {
            self.bytes_allocated[node as usize].fetch_sub(bytes, Ordering::Relaxed);
        }
    }

    /// Records a memory access.
    pub fn record_access(&self, is_local: bool) {
        if is_local {
            self.local_accesses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.remote_accesses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns the locality ratio (local / total accesses).
    pub fn locality_ratio(&self) -> f64 {
        let local = self.local_accesses.load(Ordering::Relaxed) as f64;
        let remote = self.remote_accesses.load(Ordering::Relaxed) as f64;
        
        if local + remote > 0.0 {
            local / (local + remote)
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_policy_default() {
        let policy = MemoryPolicy::default();
        assert_eq!(policy, MemoryPolicy::Local);
    }

    #[test]
    fn test_numa_stats() {
        let stats = NumaStats::default();
        
        stats.record_allocation(0, 1024);
        stats.record_access(true);
        stats.record_access(false);
        
        assert_eq!(stats.bytes_allocated[0].load(Ordering::Relaxed), 1024);
        assert_eq!(stats.locality_ratio(), 0.5);
    }
}
