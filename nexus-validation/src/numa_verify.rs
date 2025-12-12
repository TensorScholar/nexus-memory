//! NUMA Physical Page Placement Verification
//!
//! This module provides tools to verify that the `NumaAllocator` correctly
//! binds physical pages to specific NUMA nodes, ensuring the benchmark
//! results are due to actual locality optimization, not chance.
//!
//! # Verification Method
//!
//! Uses the Linux `move_pages` syscall with `nodes=NULL` to query the
//! current physical NUMA node of memory pages without moving them.
//!
//! # Requirements
//!
//! - Linux operating system
//! - Root privileges or CAP_SYS_NICE capability for `move_pages`
//! - NUMA-capable hardware (multiple nodes)
//!
//! # Example
//!
//! ```rust,ignore
//! use nexus_validation::numa_verify::{verify_physical_location, detect_numa_topology};
//!
//! if let Some(topology) = detect_numa_topology() {
//!     println!("NUMA nodes available: {:?}", topology.nodes);
//!     
//!     // Allocate memory and verify placement
//!     let ptr = allocate_on_node(1);
//!     let verified = verify_physical_location(ptr, 1);
//!     assert!(verified, "Memory not placed on expected NUMA node");
//! }
//! ```

// use std::ptr; // Removed to avoid unused warning on non-Linux
use std::fs;
use std::path::Path;

/// Result type for NUMA verification operations.
pub type Result<T> = std::result::Result<T, NumaVerifyError>;

/// Errors that can occur during NUMA verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumaVerifyError {
    /// System does not support NUMA
    NumaNotSupported,
    /// The move_pages syscall failed
    SyscallFailed(i32),
    /// Insufficient privileges for the operation
    InsufficientPrivileges,
    /// Invalid pointer provided
    InvalidPointer,
    /// The specified node does not exist
    InvalidNode(u32),
    /// Unable to read NUMA topology
    TopologyReadError,
    /// Memory is not resident (not yet paged in)
    PageNotResident,
}

impl std::fmt::Display for NumaVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumaVerifyError::NumaNotSupported => write!(f, "NUMA not supported on this system"),
            NumaVerifyError::SyscallFailed(errno) => {
                write!(f, "move_pages syscall failed with errno {}", errno)
            }
            NumaVerifyError::InsufficientPrivileges => {
                write!(f, "insufficient privileges for NUMA operations")
            }
            NumaVerifyError::InvalidPointer => write!(f, "invalid pointer provided"),
            NumaVerifyError::InvalidNode(n) => write!(f, "NUMA node {} does not exist", n),
            NumaVerifyError::TopologyReadError => write!(f, "failed to read NUMA topology"),
            NumaVerifyError::PageNotResident => write!(f, "page not resident in memory"),
        }
    }
}

impl std::error::Error for NumaVerifyError {}

/// Information about the system's NUMA topology.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// List of available NUMA node IDs
    pub nodes: Vec<u32>,
    /// Total number of NUMA nodes
    pub num_nodes: usize,
    /// Whether the system is truly NUMA (more than one node)
    pub is_numa: bool,
}

impl NumaTopology {
    /// Creates a single-node (non-NUMA) topology for fallback.
    pub fn single_node() -> Self {
        Self {
            nodes: vec![0],
            num_nodes: 1,
            is_numa: false,
        }
    }
}

/// Detects the NUMA topology of the current system.
///
/// Reads from `/sys/devices/system/node/` to discover available NUMA nodes.
///
/// # Returns
///
/// - `Some(NumaTopology)` if NUMA information could be read
/// - `None` if the system doesn't support NUMA or topology couldn't be read
pub fn detect_numa_topology() -> Option<NumaTopology> {
    let node_path = Path::new("/sys/devices/system/node");

    if !node_path.exists() {
        return None;
    }

    let entries = fs::read_dir(node_path).ok()?;

    let mut nodes = Vec::new();

    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if name_str.starts_with("node") {
            if let Ok(node_id) = name_str[4..].parse::<u32>() {
                nodes.push(node_id);
            }
        }
    }

    nodes.sort();

    let num_nodes = nodes.len();
    let is_numa = num_nodes > 1;

    Some(NumaTopology {
        nodes,
        num_nodes,
        is_numa,
    })
}

/// Gets the physical NUMA node for a memory page.
///
/// Uses the `move_pages` syscall with `nodes=NULL` to query the current
/// physical location of the page containing `ptr` without moving it.
///
/// # Arguments
///
/// * `ptr` - Pointer to the memory whose location should be queried
///
/// # Returns
///
/// - `Ok(node_id)` - The NUMA node where the page is located
/// - `Err(...)` - If the query failed
///
/// # Platform Support
///
/// This function is only available on Linux.
#[cfg(target_os = "linux")]
pub fn get_physical_node(ptr: *const u8) -> Result<i32> {
    use std::mem::MaybeUninit;

    if ptr.is_null() {
        return Err(NumaVerifyError::InvalidPointer);
    }

    // Page-align the pointer
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let page_ptr = ((ptr as usize) & !(page_size - 1)) as *mut libc::c_void;

    let mut pages: [*mut libc::c_void; 1] = [page_ptr];
    let mut status: [i32; 1] = [-1];

    // Call move_pages with nodes=NULL to query without moving
    // int move_pages(int pid, unsigned long count,
    //                void **pages, const int *nodes, int *status, int flags)
    let result = unsafe {
        libc::syscall(
            libc::SYS_move_pages,
            0i32,                    // pid = 0 means current process
            1usize,                  // count = 1 page
            pages.as_mut_ptr(),      // pages array
            std::ptr::null::<i32>(), // nodes = NULL (query only)
            status.as_mut_ptr(),     // status output
            0i32,                    // flags = 0
        )
    };

    if result != 0 {
        let errno = unsafe { *libc::__errno_location() };
        return match errno {
            libc::EPERM => Err(NumaVerifyError::InsufficientPrivileges),
            libc::ENOENT => Err(NumaVerifyError::PageNotResident),
            _ => Err(NumaVerifyError::SyscallFailed(errno)),
        };
    }

    // Check status
    let node = status[0];

    if node < 0 {
        // Negative values indicate errors
        return match -node {
            libc::ENOENT => Err(NumaVerifyError::PageNotResident),
            libc::EFAULT => Err(NumaVerifyError::InvalidPointer),
            _ => Err(NumaVerifyError::SyscallFailed(-node)),
        };
    }

    Ok(node)
}

#[cfg(not(target_os = "linux"))]
pub fn get_physical_node(_ptr: *const u8) -> Result<i32> {
    Err(NumaVerifyError::NumaNotSupported)
}

/// Verifies that a memory pointer is located on the expected NUMA node.
///
/// # Arguments
///
/// * `ptr` - Pointer to the memory to verify
/// * `expected_node` - The NUMA node where the memory should be located
///
/// # Returns
///
/// - `true` if the memory is on the expected node
/// - `false` otherwise
///
/// # Example
///
/// ```rust,ignore
/// let buffer = numa_allocator.allocate::<u64>(1024, AllocationPolicy::OnNode(NodeId(1)));
/// assert!(verify_physical_location(buffer.as_ptr() as *const u8, 1));
/// ```
pub fn verify_physical_location(ptr: *const u8, expected_node: u32) -> bool {
    match get_physical_node(ptr) {
        Ok(actual_node) => actual_node == expected_node as i32,
        Err(_) => false,
    }
}

/// Result of NUMA placement verification for a buffer.
#[derive(Debug, Clone)]
pub struct PlacementVerification {
    /// Expected NUMA node
    pub expected_node: u32,
    /// Actual NUMA node where the first page resides
    pub actual_node: Option<i32>,
    /// Whether verification succeeded
    pub verified: bool,
    /// Number of pages checked
    pub pages_checked: usize,
    /// Number of pages on the correct node
    pub pages_correct: usize,
    /// Error message if verification failed
    pub error: Option<String>,
}

/// Verifies NUMA placement for a buffer, checking multiple pages.
///
/// This function samples pages throughout the buffer to verify consistent
/// NUMA placement, which is important for large allocations.
///
/// # Arguments
///
/// * `ptr` - Start of the buffer
/// * `size` - Size of the buffer in bytes
/// * `expected_node` - Expected NUMA node
///
/// # Returns
///
/// A `PlacementVerification` with detailed results.
pub fn verify_buffer_placement(
    ptr: *const u8,
    size: usize,
    expected_node: u32,
) -> PlacementVerification {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let num_pages = (size + page_size - 1) / page_size;

    // Sample up to 100 pages evenly distributed
    let sample_count = num_pages.min(100);
    let step = if sample_count > 0 {
        num_pages / sample_count
    } else {
        1
    };

    let mut pages_checked = 0;
    let mut pages_correct = 0;
    let mut first_actual_node = None;
    let mut error = None;

    for i in (0..num_pages).step_by(step.max(1)) {
        let offset = i * page_size;
        let page_ptr = unsafe { ptr.add(offset) };

        match get_physical_node(page_ptr) {
            Ok(node) => {
                pages_checked += 1;
                if first_actual_node.is_none() {
                    first_actual_node = Some(node);
                }
                if node == expected_node as i32 {
                    pages_correct += 1;
                }
            }
            Err(e) => {
                if error.is_none() {
                    error = Some(e.to_string());
                }
            }
        }
    }

    let verified = pages_checked > 0 && pages_correct == pages_checked;

    PlacementVerification {
        expected_node,
        actual_node: first_actual_node,
        verified,
        pages_checked,
        pages_correct,
        error,
    }
}

/// Pre-faults memory to ensure physical pages are allocated.
///
/// This function touches each page in the buffer to trigger page allocation,
/// which is necessary before NUMA placement can be verified.
///
/// # Arguments
///
/// * `ptr` - Mutable pointer to the buffer
/// * `size` - Size of the buffer in bytes
///
/// # Safety
///
/// The caller must ensure `ptr` points to valid, writable memory of at least `size` bytes.
pub unsafe fn prefault_memory(ptr: *mut u8, size: usize) {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;

    for offset in (0..size).step_by(page_size) {
        // Write to each page to trigger allocation
        let page_ptr = ptr.add(offset);
        page_ptr.write_volatile(0);
    }
}

/// Pins the current thread to a specific CPU.
///
/// This is useful for FirstTouch NUMA policy, where the thread's location
/// determines memory placement.
///
/// # Arguments
///
/// * `cpu` - CPU ID to pin to
///
/// # Returns
///
/// - `Ok(())` if pinning succeeded
/// - `Err(...)` if pinning failed
#[cfg(target_os = "linux")]
pub fn pin_thread_to_cpu(cpu: usize) -> Result<()> {
    use std::mem;

    unsafe {
        let mut cpuset: libc::cpu_set_t = mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(cpu, &mut cpuset);

        let result = libc::sched_setaffinity(
            0, // current thread
            mem::size_of::<libc::cpu_set_t>(),
            &cpuset,
        );

        if result == 0 {
            Ok(())
        } else {
            Err(NumaVerifyError::SyscallFailed(*libc::__errno_location()))
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_thread_to_cpu(_cpu: usize) -> Result<()> {
    Err(NumaVerifyError::NumaNotSupported)
}

/// Gets the CPU IDs associated with a NUMA node.
///
/// Reads from `/sys/devices/system/node/nodeN/cpulist` to determine which
/// CPUs belong to the specified NUMA node.
///
/// # Arguments
///
/// * `node` - NUMA node ID
///
/// # Returns
///
/// A list of CPU IDs on the specified node, or an error.
pub fn get_cpus_on_node(node: u32) -> Result<Vec<usize>> {
    let path = format!("/sys/devices/system/node/node{}/cpulist", node);

    let content = fs::read_to_string(&path).map_err(|_| NumaVerifyError::InvalidNode(node))?;

    parse_cpu_list(&content).ok_or(NumaVerifyError::TopologyReadError)
}

/// Parses a CPU list string (e.g., "0-3,8-11") into CPU IDs.
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();

    for range in s.trim().split(',') {
        if range.contains('-') {
            let parts: Vec<&str> = range.split('-').collect();
            if parts.len() == 2 {
                let start: usize = parts[0].parse().ok()?;
                let end: usize = parts[1].parse().ok()?;
                for cpu in start..=end {
                    cpus.push(cpu);
                }
            }
        } else {
            let cpu: usize = range.parse().ok()?;
            cpus.push(cpu);
        }
    }

    Some(cpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_numa_topology() {
        // This test may pass or fail depending on the system
        if let Some(topology) = detect_numa_topology() {
            println!("NUMA topology detected:");
            println!("  Nodes: {:?}", topology.nodes);
            println!("  Is NUMA: {}", topology.is_numa);
            assert!(!topology.nodes.is_empty());
        } else {
            println!("No NUMA topology detected (single node system)");
        }
    }

    #[test]
    fn test_parse_cpu_list() {
        assert_eq!(parse_cpu_list("0-3"), Some(vec![0, 1, 2, 3]));
        assert_eq!(parse_cpu_list("0,2,4"), Some(vec![0, 2, 4]));
        assert_eq!(parse_cpu_list("0-1,4-5"), Some(vec![0, 1, 4, 5]));
        assert_eq!(parse_cpu_list("0"), Some(vec![0]));
    }

    #[test]
    fn test_single_node_topology() {
        let topo = NumaTopology::single_node();
        assert_eq!(topo.num_nodes, 1);
        assert!(!topo.is_numa);
        assert_eq!(topo.nodes, vec![0]);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_get_physical_node_stack() {
        // Test with a stack variable
        let value: u64 = 42;
        let ptr = &value as *const u64 as *const u8;

        // This may succeed or fail depending on privileges
        match get_physical_node(ptr) {
            Ok(node) => println!("Stack variable on node {}", node),
            Err(e) => println!(
                "Could not query node: {} (expected on unprivileged systems)",
                e
            ),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_verify_heap_allocation() {
        // Allocate some heap memory
        let data: Vec<u8> = vec![0u8; 4096 * 10];

        if let Some(topology) = detect_numa_topology() {
            if topology.is_numa && topology.nodes.len() >= 2 {
                // On NUMA systems, verify we can query the location
                let result = verify_buffer_placement(data.as_ptr(), data.len(), topology.nodes[0]);

                println!("Buffer placement verification: {:?}", result);
            }
        }
    }
}
