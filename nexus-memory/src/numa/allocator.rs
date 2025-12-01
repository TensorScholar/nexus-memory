//! NUMA-Aware Memory Allocator
//!
//! Provides memory allocation with explicit NUMA node placement, enabling
//! optimal data locality for cross-paradigm processing.

use core::ptr::NonNull;
use core::alloc::Layout;
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc};

use super::{NumaTopology, NodeId, MemoryPolicy, NumaStats, NumaError, Result, MAX_NUMA_NODES};

/// NUMA-aware memory allocator.
///
/// Allocates memory with explicit NUMA node placement. On systems without
/// NUMA support, falls back to standard allocation.
///
/// # Example
///
/// ```rust
/// use nexus_memory::numa::{NumaAllocator, AllocationPolicy};
///
/// let allocator = NumaAllocator::new();
///
/// // Allocate on local node
/// let ptr = allocator.allocate::<u64>(1024, AllocationPolicy::Local);
/// ```
pub struct NumaAllocator {
    /// Reference to topology
    #[cfg(feature = "std")]
    topology: &'static NumaTopology,
    
    /// Allocation statistics
    stats: NumaStats,
    
    /// Default allocation policy
    default_policy: AllocationPolicy,
}

/// Policy for memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationPolicy {
    /// Allocate on the local NUMA node
    Local,
    
    /// Allocate on a specific node
    OnNode(NodeId),
    
    /// Prefer a specific node but allow fallback
    PreferNode(NodeId),
    
    /// Interleave across all nodes (good for shared data)
    Interleaved,
    
    /// Let the OS decide (first-touch policy)
    FirstTouch,
}

impl Default for AllocationPolicy {
    fn default() -> Self {
        Self::Local
    }
}

impl NumaAllocator {
    /// Creates a new NUMA allocator.
    #[cfg(feature = "std")]
    pub fn new() -> Self {
        Self {
            topology: NumaTopology::get(),
            stats: NumaStats::default(),
            default_policy: AllocationPolicy::Local,
        }
    }

    /// Creates a new NUMA allocator with a specific default policy.
    #[cfg(feature = "std")]
    pub fn with_policy(policy: AllocationPolicy) -> Self {
        Self {
            topology: NumaTopology::get(),
            stats: NumaStats::default(),
            default_policy: policy,
        }
    }

    /// Allocates memory for `count` elements of type `T`.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of elements to allocate
    /// * `policy` - Allocation policy to use
    ///
    /// # Returns
    ///
    /// A pointer to the allocated memory, or `None` if allocation failed.
    pub fn allocate<T>(&self, count: usize, policy: AllocationPolicy) -> Option<NonNull<T>> {
        let size = count.checked_mul(core::mem::size_of::<T>())?;
        let align = core::mem::align_of::<T>().max(64); // Cache-line aligned
        
        let layout = Layout::from_size_align(size, align).ok()?;
        
        self.allocate_raw(layout, policy)
            .map(|ptr| ptr.cast())
    }

    /// Allocates raw memory with the specified layout.
    pub fn allocate_raw(&self, layout: Layout, policy: AllocationPolicy) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return NonNull::new(layout.align() as *mut u8);
        }

        let node = self.resolve_node(policy);
        
        let ptr = self.allocate_on_node(layout, node);
        
        if let Some(p) = ptr {
            self.stats.record_allocation(node.0, layout.size() as u64);
        }
        
        ptr
    }

    /// Deallocates memory.
    ///
    /// # Safety
    ///
    /// The pointer must have been allocated by this allocator with the same layout.
    pub unsafe fn deallocate<T>(&self, ptr: NonNull<T>, count: usize, node: NodeId) {
        let size = count * core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>().max(64);
        
        if let Ok(layout) = Layout::from_size_align(size, align) {
            self.deallocate_raw(ptr.cast(), layout, node);
        }
    }

    /// Deallocates raw memory.
    ///
    /// # Safety
    ///
    /// Same requirements as `deallocate`.
    pub unsafe fn deallocate_raw(&self, ptr: NonNull<u8>, layout: Layout, node: NodeId) {
        if layout.size() == 0 {
            return;
        }

        self.deallocate_from_node(ptr, layout, node);
        self.stats.record_deallocation(node.0, layout.size() as u64);
    }

    /// Returns the allocation statistics.
    pub fn stats(&self) -> &NumaStats {
        &self.stats
    }

    /// Resolves a policy to a specific node.
    fn resolve_node(&self, policy: AllocationPolicy) -> NodeId {
        match policy {
            AllocationPolicy::Local => {
                #[cfg(feature = "std")]
                {
                    self.topology.current_node()
                }
                #[cfg(not(feature = "std"))]
                {
                    NodeId(0)
                }
            }
            AllocationPolicy::OnNode(node) => node,
            AllocationPolicy::PreferNode(node) => {
                // Check if node is valid, fallback to local
                #[cfg(feature = "std")]
                {
                    if self.topology.node(node.0).is_some() {
                        node
                    } else {
                        self.topology.current_node()
                    }
                }
                #[cfg(not(feature = "std"))]
                {
                    node
                }
            }
            AllocationPolicy::Interleaved => {
                // Simple round-robin based on allocation count
                static COUNTER: AtomicU64 = AtomicU64::new(0);
                let count = COUNTER.fetch_add(1, Ordering::Relaxed);
                
                #[cfg(feature = "std")]
                {
                    let num_nodes = self.topology.num_nodes();
                    NodeId((count % num_nodes as u64) as u32)
                }
                #[cfg(not(feature = "std"))]
                {
                    NodeId(0)
                }
            }
            AllocationPolicy::FirstTouch => NodeId(0), // Let OS decide
        }
    }

    /// Platform-specific allocation on a node.
    fn allocate_on_node(&self, layout: Layout, node: NodeId) -> Option<NonNull<u8>> {
        #[cfg(target_os = "linux")]
        {
            self.allocate_linux(layout, node)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            self.allocate_fallback(layout)
        }
    }

    /// Linux-specific NUMA allocation.
    #[cfg(target_os = "linux")]
    fn allocate_linux(&self, layout: Layout, node: NodeId) -> Option<NonNull<u8>> {
        // Try libnuma if available
        // For portability, we use mmap with mbind
        
        use std::ptr;
        
        // Use mmap for large allocations, standard allocator for small ones
        if layout.size() >= 4096 {
            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    layout.size(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            
            if ptr == libc::MAP_FAILED {
                return None;
            }
            
            // Try to bind to the specified node
            // This requires libnuma, but we'll make it optional
            #[cfg(feature = "libnuma")]
            unsafe {
                let nodemask = 1u64 << node.0;
                libc::mbind(
                    ptr,
                    layout.size(),
                    libc::MPOL_BIND,
                    &nodemask as *const u64,
                    MAX_NUMA_NODES as u64,
                    0,
                );
            }
            
            NonNull::new(ptr as *mut u8)
        } else {
            // Small allocation - use standard allocator
            self.allocate_fallback(layout)
        }
    }

    /// Fallback allocation using the system allocator.
    fn allocate_fallback(&self, layout: Layout) -> Option<NonNull<u8>> {
        #[cfg(feature = "std")]
        {
            let ptr = unsafe { alloc(layout) };
            NonNull::new(ptr)
        }
        
        #[cfg(not(feature = "std"))]
        {
            None
        }
    }

    /// Platform-specific deallocation.
    fn deallocate_from_node(&self, ptr: NonNull<u8>, layout: Layout, _node: NodeId) {
        #[cfg(target_os = "linux")]
        {
            if layout.size() >= 4096 {
                unsafe {
                    libc::munmap(ptr.as_ptr() as *mut libc::c_void, layout.size());
                }
            } else {
                #[cfg(feature = "std")]
                unsafe {
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            #[cfg(feature = "std")]
            unsafe {
                dealloc(ptr.as_ptr(), layout);
            }
        }
    }
}

#[cfg(feature = "std")]
impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// A NUMA-allocated buffer.
///
/// Wraps allocated memory with its NUMA node information for proper deallocation.
pub struct NumaBuffer<T> {
    ptr: NonNull<T>,
    count: usize,
    node: NodeId,
}

impl<T> NumaBuffer<T> {
    /// Creates a new buffer on the specified node.
    pub fn new(count: usize, policy: AllocationPolicy) -> Result<Self> {
        let allocator = NumaAllocator::new();
        let node = allocator.resolve_node(policy);
        
        let ptr = allocator
            .allocate::<T>(count, policy)
            .ok_or(NumaError::AllocationFailed)?;
        
        Ok(Self { ptr, count, node })
    }

    /// Returns a pointer to the buffer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable pointer to the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the number of elements.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the NUMA node.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Returns the buffer as a slice.
    ///
    /// # Safety
    ///
    /// The buffer must have been properly initialized.
    pub unsafe fn as_slice(&self) -> &[T] {
        core::slice::from_raw_parts(self.ptr.as_ptr(), self.count)
    }

    /// Returns the buffer as a mutable slice.
    ///
    /// # Safety
    ///
    /// The buffer must have been properly initialized and caller must have
    /// exclusive access.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.count)
    }
}

impl<T> Drop for NumaBuffer<T> {
    fn drop(&mut self) {
        if self.count > 0 {
            let allocator = NumaAllocator::new();
            unsafe {
                allocator.deallocate(self.ptr, self.count, self.node);
            }
        }
    }
}

// SAFETY: NumaBuffer owns its data
unsafe impl<T: Send> Send for NumaBuffer<T> {}
unsafe impl<T: Sync> Sync for NumaBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let allocator = NumaAllocator::new();
        // Should not panic
        drop(allocator);
    }

    #[test]
    fn test_allocation() {
        let allocator = NumaAllocator::new();
        
        let ptr = allocator.allocate::<u64>(100, AllocationPolicy::Local);
        assert!(ptr.is_some());
        
        if let Some(p) = ptr {
            unsafe {
                allocator.deallocate(p, 100, NodeId(0));
            }
        }
    }

    #[test]
    fn test_numa_buffer() {
        let buffer = NumaBuffer::<u64>::new(100, AllocationPolicy::Local);
        assert!(buffer.is_ok());
        
        let buffer = buffer.unwrap();
        assert_eq!(buffer.count(), 100);
    }

    #[test]
    fn test_zero_size_allocation() {
        let allocator = NumaAllocator::new();
        
        // Zero-size allocation should work
        let ptr = allocator.allocate::<u64>(0, AllocationPolicy::Local);
        assert!(ptr.is_some());
    }

    #[test]
    fn test_policies() {
        let allocator = NumaAllocator::new();
        
        // Test different policies
        let _ = allocator.allocate::<u64>(10, AllocationPolicy::Local);
        let _ = allocator.allocate::<u64>(10, AllocationPolicy::OnNode(NodeId(0)));
        let _ = allocator.allocate::<u64>(10, AllocationPolicy::PreferNode(NodeId(0)));
        let _ = allocator.allocate::<u64>(10, AllocationPolicy::Interleaved);
        let _ = allocator.allocate::<u64>(10, AllocationPolicy::FirstTouch);
    }
}
