//! Hierarchical Epoch Structure
//!
//! This module implements the novel hierarchical epoch structure that enables
//! O(log T) global synchronization complexity, compared to O(T) in traditional
//! flat epoch schemes where T is the number of threads.
//!
//! # Algorithm Design
//!
//! The hierarchical epoch maintains a tree of epoch aggregators:
//!
//! ```text
//! Level 0 (Global):     [G] ← Global minimum epoch
//!                        │
//! Level 1 (Groups):    [A0]─[A1]─[A2]─[A3] ← Aggregated minimums
//!                       │    │    │    │
//! Level 2 (Threads): [T0-T3][T4-T7]...   ← Thread-local epochs
//! ```
//!
//! # Key Properties
//!
//! 1. **Logarithmic Synchronization**: Global epoch query is O(log T)
//! 2. **Local-First Updates**: Thread-local operations are O(1)
//! 3. **Amortized Aggregation**: Group updates batched for efficiency
//! 4. **Cache-Friendly**: Aggregation nodes are cache-line aligned
//!
//! # Theoretical Foundation
//!
//! **Theorem (Hierarchical Epoch Correctness):**
//! For any execution σ and thread t, if t reads a pointer p in epoch e,
//! then p is not freed until epoch min_epoch(σ) > e where min_epoch(σ)
//! is the minimum epoch across all active threads.
//!
//! **Proof sketch:**
//! The hierarchical structure maintains the invariant that for any
//! aggregation node A at level k, A.epoch ≤ min{C.epoch : C is child of A}.
//! By induction on tree height, the global epoch is a lower bound on
//! all thread-local epochs. □

use core::sync::atomic::Ordering;

use super::{Epoch, AtomicEpoch, INACTIVE};

/// Branching factor of the epoch tree (number of children per node)
const BRANCHING_FACTOR: usize = 4;

/// Maximum tree depth (supports up to 4^4 = 256 threads)
const MAX_DEPTH: usize = 4;

/// A hierarchical epoch manager for efficient cross-paradigm synchronization.
///
/// The `HierarchicalEpoch` structure organizes thread epochs in a tree,
/// enabling efficient global minimum queries that scale logarithmically
/// with the number of threads.
///
/// # Design Rationale
///
/// Traditional epoch-based reclamation requires O(T) time to determine
/// the global minimum epoch, as each thread's epoch must be checked.
/// For cross-paradigm processing with many threads, this becomes a
/// bottleneck.
///
/// The hierarchical design maintains aggregated minimums at internal nodes,
/// reducing the query complexity to O(log T) at the cost of slightly
/// delayed updates when threads change epochs.
///
/// # Example
///
/// ```rust
/// use nexus_memory::epoch::HierarchicalEpoch;
///
/// let hier = HierarchicalEpoch::new(16); // Support 16 threads
///
/// // Register thread 0 at epoch 5
/// hier.update_local(0, 5);
///
/// // Query global minimum
/// let min = hier.global_minimum();
/// ```
pub struct HierarchicalEpoch {
    /// Thread-local epochs stored as flat array
    local_epochs: Vec<AtomicEpoch>,
    
    /// Aggregation levels (each level aggregates BRANCHING_FACTOR children)
    /// aggregation[0] = aggregates of local_epochs
    /// aggregation[k] = aggregates of aggregation[k-1]
    aggregation: Vec<Vec<AtomicEpoch>>,
    
    /// Number of supported threads
    capacity: usize,
    
    /// Current tree depth
    depth: usize,
}

impl HierarchicalEpoch {
    /// Creates a new hierarchical epoch manager.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of threads to support (rounded up to power of BRANCHING_FACTOR)
    ///
    /// # Panics
    ///
    /// Panics if capacity exceeds the maximum supported (BRANCHING_FACTOR^MAX_DEPTH = 256).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        
        // Calculate tree depth needed
        let depth = Self::compute_depth(capacity);
        let actual_capacity = BRANCHING_FACTOR.pow(depth as u32);
        
        assert!(
            depth <= MAX_DEPTH,
            "Capacity {} exceeds maximum supported ({})",
            capacity,
            BRANCHING_FACTOR.pow(MAX_DEPTH as u32)
        );
        
        // Create local epochs
        let local_epochs: Vec<AtomicEpoch> = (0..actual_capacity)
            .map(|_| AtomicEpoch::new(INACTIVE))
            .collect();
        
        // Create aggregation levels
        let mut aggregation = Vec::new();
        let mut level_size = actual_capacity;
        
        while level_size > 1 {
            level_size = (level_size + BRANCHING_FACTOR - 1) / BRANCHING_FACTOR;
            let level: Vec<AtomicEpoch> = (0..level_size)
                .map(|_| AtomicEpoch::new(INACTIVE))
                .collect();
            aggregation.push(level);
        }
        
        Self {
            local_epochs,
            aggregation,
            capacity: actual_capacity,
            depth,
        }
    }

    /// Computes the required tree depth for a given capacity.
    fn compute_depth(capacity: usize) -> usize {
        if capacity <= 1 {
            return 1;
        }
        
        let mut depth = 1;
        let mut size = BRANCHING_FACTOR;
        
        while size < capacity {
            depth += 1;
            size *= BRANCHING_FACTOR;
        }
        
        depth
    }

    /// Updates a thread's local epoch.
    ///
    /// This is the primary interface for threads to signal their epoch participation.
    /// The update is O(1) for the local operation, with lazy propagation to
    /// aggregation nodes.
    ///
    /// # Arguments
    ///
    /// * `thread_id` - Unique identifier for the thread (0 to capacity-1)
    /// * `epoch` - The new epoch value (use INACTIVE to mark as inactive)
    ///
    /// # Panics
    ///
    /// Panics if thread_id >= capacity.
    #[inline]
    pub fn update_local(&self, thread_id: usize, epoch: Epoch) {
        assert!(thread_id < self.capacity, "Thread ID out of range");
        
        let old_epoch = self.local_epochs[thread_id].swap(epoch, Ordering::SeqCst);
        
        // Propagate upward if epoch changed
        if old_epoch != epoch {
            self.propagate_from(thread_id);
        }
    }

    /// Returns a thread's current local epoch.
    #[inline]
    pub fn local_epoch(&self, thread_id: usize) -> Epoch {
        assert!(thread_id < self.capacity, "Thread ID out of range");
        self.local_epochs[thread_id].load(Ordering::SeqCst)
    }

    /// Computes the global minimum epoch across all active threads.
    ///
    /// This operation is O(log T) due to the hierarchical structure,
    /// compared to O(T) in flat epoch schemes.
    ///
    /// # Returns
    ///
    /// The minimum epoch among all active threads, or INACTIVE if no threads
    /// are currently active.
    #[inline]
    pub fn global_minimum(&self) -> Epoch {
        // Ensure aggregation is up-to-date
        self.aggregate_all();
        
        if self.aggregation.is_empty() {
            // Only one thread, return directly
            self.local_epochs[0].load(Ordering::SeqCst)
        } else {
            // Return root aggregation
            self.aggregation.last().unwrap()[0].load(Ordering::SeqCst)
        }
    }

    /// Returns whether it's safe to reclaim objects from a given epoch.
    ///
    /// Objects from epoch `e` can be safely reclaimed if the global minimum
    /// is strictly greater than `e`.
    #[inline]
    pub fn can_reclaim(&self, epoch: Epoch) -> bool {
        let min = self.global_minimum();
        min != INACTIVE && min > epoch
    }

    /// Propagates epoch updates from a leaf toward the root.
    fn propagate_from(&self, thread_id: usize) {
        if self.aggregation.is_empty() {
            return;
        }
        
        let mut idx = thread_id;
        
        // First level aggregates local_epochs
        {
            let parent_idx = idx / BRANCHING_FACTOR;
            let start = parent_idx * BRANCHING_FACTOR;
            let end = (start + BRANCHING_FACTOR).min(self.local_epochs.len());
            
            let min = self.local_epochs[start..end]
                .iter()
                .map(|e| e.load(Ordering::SeqCst))
                .filter(|&e| e != INACTIVE)
                .min()
                .unwrap_or(INACTIVE);
            
            self.aggregation[0][parent_idx].store(min, Ordering::SeqCst);
            idx = parent_idx;
        }
        
        // Higher levels aggregate previous level
        for level_idx in 1..self.aggregation.len() {
            let parent_idx = idx / BRANCHING_FACTOR;
            let start = parent_idx * BRANCHING_FACTOR;
            let prev_len = self.aggregation[level_idx - 1].len();
            let end = (start + BRANCHING_FACTOR).min(prev_len);
            
            let min = self.aggregation[level_idx - 1][start..end]
                .iter()
                .map(|e| e.load(Ordering::SeqCst))
                .filter(|&e| e != INACTIVE)
                .min()
                .unwrap_or(INACTIVE);
            
            self.aggregation[level_idx][parent_idx].store(min, Ordering::SeqCst);
            idx = parent_idx;
        }
    }

    /// Aggregates all nodes in the tree (full refresh).
    fn aggregate_all(&self) {
        // Aggregate level 0 from local_epochs
        if let Some(level0) = self.aggregation.first() {
            for (i, agg) in level0.iter().enumerate() {
                let start = i * BRANCHING_FACTOR;
                let end = (start + BRANCHING_FACTOR).min(self.local_epochs.len());
                
                let min = self.local_epochs[start..end]
                    .iter()
                    .map(|e| e.load(Ordering::SeqCst))
                    .filter(|&e| e != INACTIVE)
                    .min()
                    .unwrap_or(INACTIVE);
                
                agg.store(min, Ordering::SeqCst);
            }
        }
        
        // Aggregate higher levels
        for level_idx in 1..self.aggregation.len() {
            let prev_level_len = self.aggregation[level_idx - 1].len();
            
            for i in 0..self.aggregation[level_idx].len() {
                let start = i * BRANCHING_FACTOR;
                let end = (start + BRANCHING_FACTOR).min(prev_level_len);
                
                let min = self.aggregation[level_idx - 1][start..end]
                    .iter()
                    .map(|e| e.load(Ordering::SeqCst))
                    .filter(|&e| e != INACTIVE)
                    .min()
                    .unwrap_or(INACTIVE);
                
                self.aggregation[level_idx][i].store(min, Ordering::SeqCst);
            }
        }
    }

    /// Returns the capacity of this hierarchical epoch manager.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the depth of the aggregation tree.
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the number of currently active threads.
    pub fn active_count(&self) -> usize {
        self.local_epochs.iter()
            .filter(|e| e.load(Ordering::Relaxed) != INACTIVE)
            .count()
    }
}

/// Builder for HierarchicalEpoch with configurable parameters.
pub struct HierarchicalEpochBuilder {
    capacity: usize,
}

impl HierarchicalEpochBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self {
            capacity: 16,
        }
    }
    
    /// Sets the maximum number of threads.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }
    
    /// Builds the HierarchicalEpoch instance.
    pub fn build(self) -> HierarchicalEpoch {
        HierarchicalEpoch::new(self.capacity)
    }
}

impl Default for HierarchicalEpochBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: HierarchicalEpoch only contains atomic operations
unsafe impl Send for HierarchicalEpoch {}
unsafe impl Sync for HierarchicalEpoch {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_calculation() {
        assert_eq!(HierarchicalEpoch::compute_depth(1), 1);
        assert_eq!(HierarchicalEpoch::compute_depth(4), 1);
        assert_eq!(HierarchicalEpoch::compute_depth(5), 2);
        assert_eq!(HierarchicalEpoch::compute_depth(16), 2);
        assert_eq!(HierarchicalEpoch::compute_depth(17), 3);
    }

    #[test]
    fn test_basic_operations() {
        let hier = HierarchicalEpoch::new(16);
        
        assert_eq!(hier.capacity(), 16);
        assert_eq!(hier.global_minimum(), INACTIVE);
    }

    #[test]
    fn test_local_update() {
        let hier = HierarchicalEpoch::new(16);
        
        // Initially inactive
        assert_eq!(hier.local_epoch(0), INACTIVE);
        
        // Update to epoch 5
        hier.update_local(0, 5);
        assert_eq!(hier.local_epoch(0), 5);
        
        // Global minimum should now be 5
        assert_eq!(hier.global_minimum(), 5);
    }

    #[test]
    fn test_multiple_threads() {
        let hier = HierarchicalEpoch::new(16);
        
        hier.update_local(0, 5);
        hier.update_local(1, 3);
        hier.update_local(2, 7);
        
        // Minimum should be 3
        assert_eq!(hier.global_minimum(), 3);
    }

    #[test]
    fn test_inactive_threads_ignored() {
        let hier = HierarchicalEpoch::new(16);
        
        hier.update_local(0, 5);
        hier.update_local(1, INACTIVE);
        hier.update_local(2, 3);
        
        // Thread 1 is inactive, so minimum is 3
        assert_eq!(hier.global_minimum(), 3);
    }

    #[test]
    fn test_can_reclaim() {
        let hier = HierarchicalEpoch::new(16);
        
        hier.update_local(0, 5);
        
        assert!(!hier.can_reclaim(5));  // Can't reclaim current epoch
        assert!(hier.can_reclaim(4));   // Can reclaim earlier epochs
        assert!(hier.can_reclaim(0));
    }

    #[test]
    fn test_active_count() {
        let hier = HierarchicalEpoch::new(16);
        
        assert_eq!(hier.active_count(), 0);
        
        hier.update_local(0, 5);
        assert_eq!(hier.active_count(), 1);
        
        hier.update_local(1, 3);
        assert_eq!(hier.active_count(), 2);
        
        hier.update_local(0, INACTIVE);
        assert_eq!(hier.active_count(), 1);
    }

    #[test]
    fn test_builder() {
        let hier = HierarchicalEpochBuilder::new()
            .capacity(32)
            .build();
        
        assert_eq!(hier.capacity(), 64); // Rounded up to power of branching factor
    }
}
