//! NUMA Topology Discovery
//!
//! Discovers and represents the NUMA topology of the system, including
//! node count, CPU affinity, memory capacity, and inter-node distances.

use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::{NumaNode, NumaError, Result, MAX_NUMA_NODES};

/// Global topology instance
#[cfg(feature = "std")]
static TOPOLOGY: OnceLock<NumaTopology> = OnceLock::new();

/// A set of CPU IDs.
#[derive(Debug, Clone, Default)]
pub struct CpuSet {
    /// Bitmask of CPUs (supports up to 256 CPUs)
    mask: [u64; 4],
}

impl CpuSet {
    /// Creates an empty CPU set.
    pub const fn new() -> Self {
        Self { mask: [0; 4] }
    }

    /// Adds a CPU to the set.
    pub fn insert(&mut self, cpu: u32) {
        if cpu < 256 {
            let idx = (cpu / 64) as usize;
            let bit = cpu % 64;
            self.mask[idx] |= 1 << bit;
        }
    }

    /// Removes a CPU from the set.
    pub fn remove(&mut self, cpu: u32) {
        if cpu < 256 {
            let idx = (cpu / 64) as usize;
            let bit = cpu % 64;
            self.mask[idx] &= !(1 << bit);
        }
    }

    /// Checks if a CPU is in the set.
    pub fn contains(&self, cpu: u32) -> bool {
        if cpu < 256 {
            let idx = (cpu / 64) as usize;
            let bit = cpu % 64;
            (self.mask[idx] & (1 << bit)) != 0
        } else {
            false
        }
    }

    /// Returns the number of CPUs in the set.
    pub fn count(&self) -> usize {
        self.mask.iter().map(|m| m.count_ones() as usize).sum()
    }

    /// Returns an iterator over the CPUs in the set.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..256u32).filter(|&cpu| self.contains(cpu))
    }

    /// Returns whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.mask.iter().all(|&m| m == 0)
    }
}

/// A NUMA node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Creates a new node ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw ID.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for NodeId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// NUMA topology information.
///
/// Provides information about the system's NUMA topology, including
/// the number of nodes, their distances, and associated CPUs.
#[derive(Debug)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    num_nodes: usize,
    
    /// Node information
    nodes: Vec<NumaNode>,
    
    /// Whether NUMA is actually available
    numa_available: bool,
}

impl NumaTopology {
    /// Gets or initializes the global topology.
    #[cfg(feature = "std")]
    pub fn get() -> &'static Self {
        TOPOLOGY.get_or_init(|| {
            Self::discover().unwrap_or_else(|_| Self::fallback())
        })
    }

    /// Discovers the system's NUMA topology.
    pub fn discover() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::discover_linux()
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS doesn't expose NUMA topology
            Ok(Self::fallback())
        }
        
        #[cfg(target_os = "windows")]
        {
            Self::discover_windows()
        }
        
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            Ok(Self::fallback())
        }
    }

    /// Creates a fallback single-node topology.
    pub fn fallback() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        
        let mut cpus = CpuSet::new();
        for i in 0..num_cpus {
            cpus.insert(i as u32);
        }
        
        // Estimate memory (use 16 GB as default)
        let memory = 16 * 1024 * 1024 * 1024u64;
        
        let mut distances = [255u8; MAX_NUMA_NODES];
        distances[0] = 10; // Local distance
        
        let node = NumaNode::new(0, cpus, memory, distances);
        
        Self {
            num_nodes: 1,
            nodes: vec![node],
            numa_available: false,
        }
    }

    /// Linux-specific topology discovery.
    #[cfg(target_os = "linux")]
    fn discover_linux() -> Result<Self> {
        use std::fs;
        use std::path::Path;
        
        let numa_path = Path::new("/sys/devices/system/node");
        
        if !numa_path.exists() {
            return Ok(Self::fallback());
        }
        
        // Find all node directories
        let mut node_ids: Vec<u32> = Vec::new();
        
        if let Ok(entries) = fs::read_dir(numa_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                
                if name_str.starts_with("node") {
                    if let Ok(id) = name_str[4..].parse::<u32>() {
                        node_ids.push(id);
                    }
                }
            }
        }
        
        if node_ids.is_empty() {
            return Ok(Self::fallback());
        }
        
        node_ids.sort();
        
        let mut nodes = Vec::with_capacity(node_ids.len());
        
        for &node_id in &node_ids {
            let node_path = numa_path.join(format!("node{}", node_id));
            
            // Read CPU list
            let mut cpus = CpuSet::new();
            if let Ok(cpu_list) = fs::read_to_string(node_path.join("cpulist")) {
                for range in cpu_list.trim().split(',') {
                    if let Some(dash) = range.find('-') {
                        if let (Ok(start), Ok(end)) = (
                            range[..dash].parse::<u32>(),
                            range[dash + 1..].parse::<u32>(),
                        ) {
                            for cpu in start..=end {
                                cpus.insert(cpu);
                            }
                        }
                    } else if let Ok(cpu) = range.parse::<u32>() {
                        cpus.insert(cpu);
                    }
                }
            }
            
            // Read memory info
            let memory = fs::read_to_string(node_path.join("meminfo"))
                .ok()
                .and_then(|content| {
                    for line in content.lines() {
                        if line.contains("MemTotal:") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 4 {
                                if let Ok(kb) = parts[3].parse::<u64>() {
                                    return Some(kb * 1024);
                                }
                            }
                        }
                    }
                    None
                })
                .unwrap_or(0);
            
            // Read distances
            let mut distances = [255u8; MAX_NUMA_NODES];
            if let Ok(dist_str) = fs::read_to_string(node_path.join("distance")) {
                for (i, dist) in dist_str.trim().split_whitespace().enumerate() {
                    if i < MAX_NUMA_NODES {
                        if let Ok(d) = dist.parse::<u8>() {
                            distances[i] = d;
                        }
                    }
                }
            }
            
            nodes.push(NumaNode::new(node_id, cpus, memory, distances));
        }
        
        Ok(Self {
            num_nodes: nodes.len(),
            nodes,
            numa_available: true,
        })
    }

    /// Windows-specific topology discovery.
    #[cfg(target_os = "windows")]
    fn discover_windows() -> Result<Self> {
        // Simplified Windows implementation
        Ok(Self::fallback())
    }

    /// Returns the number of NUMA nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns whether NUMA is available.
    #[inline]
    pub fn is_numa_available(&self) -> bool {
        self.numa_available
    }

    /// Gets a node by ID.
    pub fn node(&self, id: u32) -> Option<&NumaNode> {
        self.nodes.iter().find(|n| n.id() == id)
    }

    /// Returns an iterator over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &NumaNode> {
        self.nodes.iter()
    }

    /// Gets the current node for the calling thread.
    pub fn current_node(&self) -> NodeId {
        #[cfg(target_os = "linux")]
        {
            // Use getcpu if available
            let cpu = unsafe { libc::sched_getcpu() };
            if cpu >= 0 {
                for node in &self.nodes {
                    if node.cpus().contains(cpu as u32) {
                        return NodeId(node.id());
                    }
                }
            }
        }
        
        // Default to node 0
        NodeId(0)
    }

    /// Gets the distance between two nodes.
    pub fn distance(&self, from: NodeId, to: NodeId) -> u8 {
        self.node(from.0)
            .map(|n| n.distance_to(to.0))
            .unwrap_or(255)
    }

    /// Finds the optimal node for a given set of accessing threads.
    ///
    /// Returns the node that minimizes total access distance.
    pub fn optimal_node(&self, thread_nodes: &[NodeId]) -> NodeId {
        if thread_nodes.is_empty() || self.num_nodes == 1 {
            return NodeId(0);
        }
        
        let mut best_node = NodeId(0);
        let mut best_total_distance = u64::MAX;
        
        for node in &self.nodes {
            let total_distance: u64 = thread_nodes
                .iter()
                .map(|&tn| self.distance(NodeId(node.id()), tn) as u64)
                .sum();
            
            if total_distance < best_total_distance {
                best_total_distance = total_distance;
                best_node = NodeId(node.id());
            }
        }
        
        best_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_set() {
        let mut set = CpuSet::new();
        
        assert!(set.is_empty());
        
        set.insert(0);
        set.insert(5);
        set.insert(100);
        
        assert!(set.contains(0));
        assert!(set.contains(5));
        assert!(set.contains(100));
        assert!(!set.contains(1));
        
        assert_eq!(set.count(), 3);
        
        set.remove(5);
        assert!(!set.contains(5));
        assert_eq!(set.count(), 2);
    }

    #[test]
    fn test_fallback_topology() {
        let topology = NumaTopology::fallback();
        
        assert_eq!(topology.num_nodes(), 1);
        assert!(!topology.is_numa_available());
        
        let node = topology.node(0).unwrap();
        assert_eq!(node.id(), 0);
        assert!(node.cpus().count() > 0);
    }

    #[test]
    fn test_node_id() {
        let id = NodeId::new(5);
        assert_eq!(id.as_u32(), 5);
        
        let id2: NodeId = 10u32.into();
        assert_eq!(id2.as_u32(), 10);
    }

    #[test]
    fn test_distance() {
        let topology = NumaTopology::fallback();
        
        // Self-distance should be 10 (local)
        assert_eq!(topology.distance(NodeId(0), NodeId(0)), 10);
    }
}
