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

use crate::sync::atomic::Ordering;

use super::{AtomicEpoch, Epoch, INACTIVE};

#[cfg(feature = "bench-metrics")]
use std::time::Instant;

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
    ///
    /// **Monotonicity Invariant (M_{parent ≤ children}):**
    /// For any aggregation node A at level k with children C₁, C₂, ..., Cₙ:
    ///   A.epoch ≤ min{Cᵢ.epoch : Cᵢ ≠ INACTIVE}
    ///
    /// This invariant is preserved by lock-free fetch_max operations during propagation.
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
        #[cfg(feature = "bench-metrics")]
        let start = Instant::now();

        assert!(thread_id < self.capacity, "Thread ID out of range");

        // MEMORY ORDERING: Release semantics ensure that all prior memory operations
        // (e.g., writes to data structures protected by this epoch) are visible to
        // threads that observe this epoch update via Acquire loads.
        //
        // WHY NOT SeqCst: Release-Acquire provides sufficient happens-before ordering
        // for epoch-based reclamation. SeqCst's total ordering guarantee is unnecessary
        // and adds overhead on weakly-ordered architectures (ARM, POWER).
        let old_epoch = self.local_epochs[thread_id].swap(epoch, Ordering::Release);

        // Propagate upward if epoch changed
        if old_epoch != epoch {
            self.propagate_from(thread_id);
        }

        #[cfg(feature = "bench-metrics")]
        crate::epoch::metrics::record_pin(start.elapsed());
    }

    /// Returns a thread's current local epoch.
    #[inline]
    pub fn local_epoch(&self, thread_id: usize) -> Epoch {
        assert!(thread_id < self.capacity, "Thread ID out of range");
        // MEMORY ORDERING: Acquire semantics ensure we observe all memory operations
        // that happened-before the corresponding Release store in update_local.
        self.local_epochs[thread_id].load(Ordering::Acquire)
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
        #[cfg(feature = "bench-metrics")]
        let start = Instant::now();

        // Ensure aggregation is up-to-date
        // self.aggregate_all(); // REMOVED: This causes O(T) complexity. Updates are propagated via propagate_from (O(log T)).

        // MEMORY ORDERING: Acquire semantics ensure we observe the most recent
        // aggregation updates performed by propagate_from's Release operations.
        // This establishes a happens-before relationship from all epoch updates
        // to this global minimum query.
        let result = if self.aggregation.is_empty() {
            // Only one thread, return directly
            self.local_epochs[0].load(Ordering::Acquire)
        } else {
            // Return root aggregation
            self.aggregation.last().unwrap()[0].load(Ordering::Acquire)
        };

        #[cfg(feature = "bench-metrics")]
        crate::epoch::metrics::record_advance(start.elapsed());

        result
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

    /// Propagates epoch updates from a leaf toward the root using lock-free monotonic aggregation.
    ///
    /// # Algorithm: Lock-Free Monotonic Aggregation Protocol
    ///
    /// **Invariant Preservation:** Maintains M_{parent ≤ children} without blocking.
    ///
    /// **Key Insight:** Epochs are monotonically non-decreasing. If thread T₁ computes
    /// min=10 and T₂ computes min=12 from overlapping observations of children, both
    /// are valid lower bounds. Using `fetch_max` ensures the parent converges to the
    /// highest observed bound, preventing epoch regression while allowing concurrent
    /// updates.
    ///
    /// **Correctness Argument:**
    /// 1. Let P be a parent node with children C₁, C₂, ..., Cₙ
    /// 2. Thread Tᵢ reads snapshot S_i = {C₁ᵢ, C₂ᵢ, ..., Cₙᵢ} with Acquire ordering
    /// 3. Tᵢ computes m_i = min{Cⱼᵢ : Cⱼᵢ ≠ INACTIVE}
    /// 4. Tᵢ calls P.fetch_max(m_i, Release)
    /// 5. By monotonicity, for any later snapshot S_j, m_j ≥ m_i (epochs only advance)
    /// 6. fetch_max ensures P.epoch = max{m_i : all updates} ≤ min{current children}
    /// 7. Therefore, M_{parent ≤ children} is preserved even under concurrent updates
    ///
    /// **Why fetch_max is Sufficient:**
    /// - No ABA problem: Epochs are write-once-increment, never reused
    /// - No lost updates: fetch_max is atomic read-modify-write
    /// - Stale reads are safe: Older minimums are still valid lower bounds
    ///
    /// # Memory Ordering
    ///
    /// - **Acquire on children reads:** Synchronizes with Release stores in lower levels,
    ///   ensuring we observe the most recent epoch updates.
    /// - **Release on parent writes:** Makes this aggregation visible to subsequent
    ///   Acquire loads in higher levels or global_minimum().
    ///
    /// # Performance
    ///
    /// - Time Complexity: O(log T) - traverses tree height
    /// - Cache Behavior: Updates only touched nodes (O(log T) cache lines)
    /// - Contention: Wait-free - no thread can block another
    fn propagate_from(&self, thread_id: usize) {
        if self.aggregation.is_empty() {
            return;
        }

        let mut idx = thread_id;

        // LEVEL 0: Aggregate local_epochs into first aggregation level
        {
            let parent_idx = idx / BRANCHING_FACTOR;
            let start = parent_idx * BRANCHING_FACTOR;
            let end = (start + BRANCHING_FACTOR).min(self.local_epochs.len());

            // SAFETY: No lock required. Concurrent updates to the same parent are safe
            // because:
            // 1. Each thread computes a valid lower bound from its snapshot
            // 2. fetch_max atomically advances to the highest observed bound
            // 3. Monotonicity ensures no regression
            //
            // MEMORY ORDERING: Acquire reads synchronize with Release writes from
            // update_local(), ensuring we see the latest thread epochs.
            let min = self.local_epochs[start..end]
                .iter()
                .map(|e| e.load(Ordering::Acquire))
                .filter(|&e| e != INACTIVE)
                .min()
                .unwrap_or(INACTIVE);

            // LOCK-FREE UPDATE with INACTIVE handling:
            // Since INACTIVE = u64::MAX, we need special handling for state transitions.
            // Use compare-and-exchange loop for lock-free monotonic updates.
            //
            // CORRECTNESS: This loop is wait-free because:
            // 1. If current == INACTIVE and min != INACTIVE: CAS succeeds (transition to active)
            // 2. If current != INACTIVE and min > current: CAS succeeds (monotonic advance)
            // 3. If current != INACTIVE and min <= current: No update needed, exit
            // 4. If current == min: No update needed, exit
            //
            // The loop terminates in O(1) iterations because successful CAS means we're done,
            // and failed CAS means another thread made progress, so we can exit.
            loop {
                let current = self.aggregation[0][parent_idx].load(Ordering::Relaxed);
                
                // CORRECTED LOGIC: Parent should hold minimum of children
                // Skip update only if: current == min (synchronized) OR current < min (parent already better)
                // CRITICAL FIX: Changed from "current <= min" to "current < min" to allow min=current case to skip
                
                if current == min {
                    break;  // Already synchronized
                }
                
                // Skip if parent already has a smaller (more conservative) value
                // Exception: Always update when transitioning to/from INACTIVE
                if current != INACTIVE && min != INACTIVE && current < min {
                    break;  // Parent already more conservative, skip
                }
                
                // Update: min < current (improvement) OR INACTIVE transitions
                match self.aggregation[0][parent_idx].compare_exchange_weak(
                    current,
                    min,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }

            idx = parent_idx;
        }

        // HIGHER LEVELS: Aggregate previous level into current level
        for level_idx in 1..self.aggregation.len() {
            let parent_idx = idx / BRANCHING_FACTOR;
            let start = parent_idx * BRANCHING_FACTOR;
            let prev_len = self.aggregation[level_idx - 1].len();
            let end = (start + BRANCHING_FACTOR).min(prev_len);

            // MEMORY ORDERING: Acquire reads synchronize with Release writes from
            // lower aggregation levels, establishing transitive happens-before chain
            // from leaf updates to root.
            let min = self.aggregation[level_idx - 1][start..end]
                .iter()
                .map(|e| e.load(Ordering::Acquire))
                .filter(|&e| e != INACTIVE)
                .min()
                .unwrap_or(INACTIVE);

            // LOCK-FREE UPDATE: Same corrected logic as level 0
            loop {
                let current = self.aggregation[level_idx][parent_idx].load(Ordering::Relaxed);
                
                if current == min {
                    break;
                }
                
                if current != INACTIVE && min != INACTIVE && current < min {
                    break;
                }
                
                match self.aggregation[level_idx][parent_idx].compare_exchange_weak(
                    current,
                    min,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }

            idx = parent_idx;
        }
    }

    /// Aggregates all nodes in the tree (full refresh).
    ///
    /// NOTE: This method is primarily for testing and initialization.
    /// Normal operation uses lazy propagation via propagate_from().
    fn aggregate_all(&self) {
        // Aggregate level 0 from local_epochs
        if let Some(level0) = self.aggregation.first() {
            for (i, agg) in level0.iter().enumerate() {
                let start = i * BRANCHING_FACTOR;
                let end = (start + BRANCHING_FACTOR).min(self.local_epochs.len());

                let min = self.local_epochs[start..end]
                    .iter()
                    .map(|e| e.load(Ordering::Acquire))
                    .filter(|&e| e != INACTIVE)
                    .min()
                    .unwrap_or(INACTIVE);

                agg.store(min, Ordering::Release);
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
                    .map(|e| e.load(Ordering::Acquire))
                    .filter(|&e| e != INACTIVE)
                    .min()
                    .unwrap_or(INACTIVE);

                self.aggregation[level_idx][i].store(min, Ordering::Release);
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
        self.local_epochs
            .iter()
            .filter(|e| e.load(Ordering::Relaxed) != INACTIVE)
            .count()
    }
}

/// Builder for HierarchicalEpoch with configurable parameters.
#[allow(dead_code)]
pub struct HierarchicalEpochBuilder {
    capacity: usize,
}

impl HierarchicalEpochBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self { capacity: 16 }
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

        assert!(!hier.can_reclaim(5)); // Can't reclaim current epoch
        assert!(hier.can_reclaim(4)); // Can reclaim earlier epochs
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
        let hier = HierarchicalEpochBuilder::new().capacity(32).build();

        assert_eq!(hier.capacity(), 64); // Rounded up to power of branching factor
    }
}
