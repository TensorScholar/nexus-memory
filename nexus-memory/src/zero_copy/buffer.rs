//! Zero-Copy Buffer Implementation
//!
//! Provides the core buffer type that enables zero-copy data sharing across
//! computational paradigms. The buffer maintains a single memory region that
//! can be safely accessed from batch, stream, and graph processing code.
//!
//! # Memory Layout
//!
//! The buffer uses cache-line aligned allocation for optimal performance:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes)                                           │
//! │ ├─ capacity: usize                                          │
//! │ ├─ len: AtomicUsize                                         │
//! │ ├─ readers: AtomicU32                                       │
//! │ ├─ writer: AtomicBool                                       │
//! │ └─ [padding]                                                │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Data Region (cache-line aligned)                            │
//! │ [T; capacity]                                               │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Thread Safety
//!
//! The buffer uses atomic operations for borrow tracking, allowing multiple
//! readers or a single writer at any time (similar to RwLock but without blocking).

use core::sync::atomic::{AtomicU32, AtomicUsize, AtomicBool, Ordering};
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::mem::{self, MaybeUninit};
use core::ops::{Deref, DerefMut};
use core::slice;

#[cfg(not(feature = "std"))]
use alloc::{alloc::{alloc, dealloc, Layout}, boxed::Box, vec::Vec};

#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc, Layout};

use super::{ZeroCopyError, Result, Region, ZeroCopyRef, ZeroCopyMut, PhantomLifetime};

/// Cache line size for alignment
const CACHE_LINE: usize = 64;

/// A zero-copy buffer for cross-paradigm data sharing.
///
/// `ZeroCopyBuffer<T>` provides a contiguous memory region that can be accessed
/// by different processing paradigms without copying data. The buffer tracks
/// borrows at runtime to ensure memory safety.
///
/// # Type Parameters
///
/// - `T`: The element type. Must be `Send` for thread safety.
///
/// # Example
///
/// ```rust
/// use nexus_memory::zero_copy::ZeroCopyBuffer;
///
/// let mut buffer = ZeroCopyBuffer::<u64>::new(1000);
///
/// // Write some data
/// buffer.push(42);
/// buffer.push(100);
///
/// // Read back
/// assert_eq!(buffer.get(0), Some(&42));
/// assert_eq!(buffer.len(), 2);
/// ```
pub struct ZeroCopyBuffer<T> {
    /// Pointer to the data region
    data: NonNull<T>,
    
    /// Total capacity in elements
    capacity: usize,
    
    /// Current number of valid elements
    len: AtomicUsize,
    
    /// Number of active readers
    readers: AtomicU32,
    
    /// Whether a writer is active
    writer: AtomicBool,
    
    /// Marker for ownership semantics
    _marker: PhantomData<T>,
}

// SAFETY: ZeroCopyBuffer manages its own synchronization
unsafe impl<T: Send> Send for ZeroCopyBuffer<T> {}
unsafe impl<T: Send + Sync> Sync for ZeroCopyBuffer<T> {}

impl<T> ZeroCopyBuffer<T> {
    /// Creates a new buffer with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of elements the buffer can hold
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or if `capacity * size_of::<T>()` overflows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use nexus_memory::zero_copy::ZeroCopyBuffer;
    ///
    /// let buffer = ZeroCopyBuffer::<i32>::new(100);
    /// assert_eq!(buffer.capacity(), 100);
    /// assert_eq!(buffer.len(), 0);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self::try_new(capacity).expect("Failed to allocate ZeroCopyBuffer")
    }

    /// Tries to create a new buffer, returning an error on failure.
    pub fn try_new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                capacity: 0,
                len: AtomicUsize::new(0),
                readers: AtomicU32::new(0),
                writer: AtomicBool::new(false),
                _marker: PhantomData,
            });
        }

        let size = capacity
            .checked_mul(mem::size_of::<T>())
            .ok_or(ZeroCopyError::AllocationFailed)?;
        
        let align = mem::align_of::<T>().max(CACHE_LINE);
        
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| ZeroCopyError::InvalidAlignment)?;
        
        // SAFETY: layout is valid and non-zero
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(ZeroCopyError::AllocationFailed);
        }
        
        Ok(Self {
            data: NonNull::new(ptr as *mut T).unwrap(),
            capacity,
            len: AtomicUsize::new(0),
            readers: AtomicU32::new(0),
            writer: AtomicBool::new(false),
            _marker: PhantomData,
        })
    }

    /// Returns the buffer's capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the current number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Returns whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Returns a raw pointer to the data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable raw pointer to the data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }

    /// Gets an element by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            // SAFETY: index is bounds-checked
            Some(unsafe { &*self.data.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// Gets a mutable element by index.
    ///
    /// # Safety
    ///
    /// Caller must ensure no other references to this element exist.
    #[inline]
    pub unsafe fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            // SAFETY: index is bounds-checked and caller ensures exclusivity
            Some(unsafe { &mut *self.data.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// Pushes an element to the end of the buffer.
    ///
    /// # Returns
    ///
    /// `Ok(index)` where `index` is the position of the new element,
    /// or `Err` if the buffer is full.
    pub fn push(&self, value: T) -> Result<usize> {
        loop {
            let current_len = self.len.load(Ordering::Acquire);
            
            if current_len >= self.capacity {
                return Err(ZeroCopyError::InsufficientCapacity {
                    required: current_len + 1,
                    available: self.capacity,
                });
            }
            
            // Try to claim the slot
            match self.len.compare_exchange_weak(
                current_len,
                current_len + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: We've claimed the slot
                    unsafe {
                        self.data.as_ptr().add(current_len).write(value);
                    }
                    return Ok(current_len);
                }
                Err(_) => {
                    // Retry
                    core::hint::spin_loop();
                }
            }
        }
    }

    /// Writes a slice of data starting at the specified offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting position in the buffer
    /// * `data` - Slice of data to write
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive write access to the affected region.
    pub unsafe fn write(&self, offset: usize, data: &[T]) 
    where
        T: Copy,
    {
        assert!(offset + data.len() <= self.capacity, "Write exceeds capacity");
        
        // SAFETY: Caller ensures exclusive access
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data.as_ptr().add(offset),
                data.len(),
            );
        }
        
        // Update length if we've extended it
        loop {
            let current = self.len.load(Ordering::Acquire);
            let new_len = (offset + data.len()).max(current);
            
            if current == new_len {
                break;
            }
            
            match self.len.compare_exchange_weak(
                current, new_len, Ordering::AcqRel, Ordering::Acquire
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    /// Returns an immutable zero-copy reference to the buffer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a writer is currently active.
    pub fn try_borrow(&self) -> Result<BufferRef<'_, T>> {
        // Check if a writer is active
        if self.writer.load(Ordering::Acquire) {
            return Err(ZeroCopyError::BorrowConflict);
        }
        
        // Increment reader count
        self.readers.fetch_add(1, Ordering::AcqRel);
        
        // Double-check no writer became active
        if self.writer.load(Ordering::Acquire) {
            self.readers.fetch_sub(1, Ordering::AcqRel);
            return Err(ZeroCopyError::BorrowConflict);
        }
        
        Ok(BufferRef {
            buffer: self,
            _marker: PhantomData,
        })
    }

    /// Returns a mutable zero-copy reference to the buffer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any readers or another writer is active.
    pub fn try_borrow_mut(&self) -> Result<BufferMut<'_, T>> {
        // Try to set writer flag
        if self.writer.compare_exchange(
            false, true, Ordering::AcqRel, Ordering::Acquire
        ).is_err() {
            return Err(ZeroCopyError::BorrowConflict);
        }
        
        // Check for active readers
        if self.readers.load(Ordering::Acquire) > 0 {
            self.writer.store(false, Ordering::Release);
            return Err(ZeroCopyError::BorrowConflict);
        }
        
        Ok(BufferMut {
            buffer: self,
            _marker: PhantomData,
        })
    }

    /// Returns a reference to the underlying data as a slice.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent modifications.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[T] {
        // SAFETY: Caller ensures safety
        unsafe {
            slice::from_raw_parts(self.data.as_ptr(), self.len())
        }
    }

    /// Returns a mutable reference to the underlying data as a slice.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive access.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len();
        // SAFETY: We have &mut self
        unsafe {
            slice::from_raw_parts_mut(self.data.as_ptr(), len)
        }
    }

    /// Clears the buffer, resetting the length to zero.
    ///
    /// # Safety
    ///
    /// Caller must ensure exclusive access and that no references exist.
    pub unsafe fn clear(&self) {
        // Drop existing elements if T needs dropping
        if mem::needs_drop::<T>() {
            let len = self.len.load(Ordering::Acquire);
            for i in 0..len {
                unsafe {
                    core::ptr::drop_in_place(self.data.as_ptr().add(i));
                }
            }
        }
        
        self.len.store(0, Ordering::Release);
    }

    /// Creates a view into a region of the buffer.
    pub fn region(&self, region: Region) -> Result<BufferRegion<'_, T>> {
        let len = self.len();
        
        if region.end() > len {
            return Err(ZeroCopyError::OutOfBounds {
                index: region.end(),
                len,
            });
        }
        
        // Try to borrow
        let _ref = self.try_borrow()?;
        
        Ok(BufferRegion {
            buffer: self,
            region,
            _marker: PhantomData,
        })
    }
}

impl<T> Drop for ZeroCopyBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // Drop elements
            // SAFETY: We have exclusive access in Drop
            unsafe { self.clear() };
            
            // Deallocate memory
            let size = self.capacity * mem::size_of::<T>();
            let align = mem::align_of::<T>().max(CACHE_LINE);
            
            if let Ok(layout) = Layout::from_size_align(size, align) {
                // SAFETY: layout matches original allocation
                unsafe {
                    dealloc(self.data.as_ptr() as *mut u8, layout);
                }
            }
        }
    }
}

/// Immutable borrow of a zero-copy buffer.
pub struct BufferRef<'a, T> {
    buffer: &'a ZeroCopyBuffer<T>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> BufferRef<'a, T> {
    /// Returns the length of the borrowed region.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns whether the borrowed region is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets an element by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.buffer.get(index)
    }

    /// Returns the borrowed data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: We hold a read borrow
        unsafe { self.buffer.as_slice() }
    }
}

impl<T> Drop for BufferRef<'_, T> {
    fn drop(&mut self) {
        self.buffer.readers.fetch_sub(1, Ordering::AcqRel);
    }
}

impl<T> Deref for BufferRef<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Mutable borrow of a zero-copy buffer.
pub struct BufferMut<'a, T> {
    buffer: &'a ZeroCopyBuffer<T>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> BufferMut<'a, T> {
    /// Returns the length of the borrowed region.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns whether the borrowed region is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets a mutable element by index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            // SAFETY: We have exclusive write access
            Some(unsafe { &mut *self.buffer.data.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// Returns the borrowed data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len();
        // SAFETY: We have exclusive access
        unsafe {
            slice::from_raw_parts_mut(self.buffer.data.as_ptr(), len)
        }
    }
}

impl<T> Drop for BufferMut<'_, T> {
    fn drop(&mut self) {
        self.buffer.writer.store(false, Ordering::Release);
    }
}

impl<T> Deref for BufferMut<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // SAFETY: We have exclusive access
        unsafe { self.buffer.as_slice() }
    }
}

impl<T> DerefMut for BufferMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// A view into a region of a zero-copy buffer.
pub struct BufferRegion<'a, T> {
    buffer: &'a ZeroCopyBuffer<T>,
    region: Region,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> BufferRegion<'a, T> {
    /// Returns the region specification.
    pub fn region(&self) -> Region {
        self.region
    }

    /// Returns the length of this region.
    pub fn len(&self) -> usize {
        self.region.len
    }

    /// Returns whether this region is empty.
    pub fn is_empty(&self) -> bool {
        self.region.len == 0
    }

    /// Gets an element by index within the region.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.region.len {
            self.buffer.get(self.region.start + index)
        } else {
            None
        }
    }

    /// Returns the region as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: Region bounds were checked during construction
        unsafe {
            slice::from_raw_parts(
                self.buffer.data.as_ptr().add(self.region.start),
                self.region.len,
            )
        }
    }
}

impl<T> Deref for BufferRegion<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Builder for customizing buffer allocation.
pub struct ZeroCopyBufferBuilder {
    capacity: usize,
    alignment: usize,
}

impl ZeroCopyBufferBuilder {
    /// Creates a new buffer builder.
    pub fn new() -> Self {
        Self {
            capacity: 0,
            alignment: CACHE_LINE,
        }
    }

    /// Sets the buffer capacity.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets custom alignment (must be power of 2).
    pub fn alignment(mut self, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        self.alignment = alignment;
        self
    }

    /// Builds the buffer.
    pub fn build<T>(self) -> Result<ZeroCopyBuffer<T>> {
        ZeroCopyBuffer::try_new(self.capacity)
    }
}

impl Default for ZeroCopyBufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffer = ZeroCopyBuffer::<i32>::new(100);
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_push() {
        let buffer = ZeroCopyBuffer::<i32>::new(10);
        
        buffer.push(1).unwrap();
        buffer.push(2).unwrap();
        buffer.push(3).unwrap();
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&3));
    }

    #[test]
    fn test_full_buffer() {
        let buffer = ZeroCopyBuffer::<i32>::new(2);
        
        buffer.push(1).unwrap();
        buffer.push(2).unwrap();
        
        let result = buffer.push(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_borrow() {
        let buffer = ZeroCopyBuffer::<i32>::new(10);
        buffer.push(42).unwrap();
        
        let ref1 = buffer.try_borrow().unwrap();
        let ref2 = buffer.try_borrow().unwrap();
        
        assert_eq!(ref1.get(0), Some(&42));
        assert_eq!(ref2.get(0), Some(&42));
    }

    #[test]
    fn test_borrow_mut_conflict() {
        let buffer = ZeroCopyBuffer::<i32>::new(10);
        buffer.push(42).unwrap();
        
        let _ref1 = buffer.try_borrow().unwrap();
        let result = buffer.try_borrow_mut();
        
        assert!(result.is_err());
    }

    #[test]
    fn test_region() {
        let buffer = ZeroCopyBuffer::<i32>::new(100);
        
        for i in 0..50 {
            buffer.push(i).unwrap();
        }
        
        let region = buffer.region(Region::new(10, 20)).unwrap();
        assert_eq!(region.len(), 20);
        assert_eq!(region.get(0), Some(&10));
        assert_eq!(region.get(19), Some(&29));
    }

    #[test]
    fn test_builder() {
        let buffer: ZeroCopyBuffer<u64> = ZeroCopyBufferBuilder::new()
            .capacity(50)
            .build()
            .unwrap();
        
        assert_eq!(buffer.capacity(), 50);
    }
}
