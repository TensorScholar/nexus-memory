//! Phantom Lifetime Tracking for Zero-Copy References
//!
//! This module implements compile-time lifetime tracking for zero-copy references
//! using Rust's type system. The key insight is that cross-paradigm data sharing
//! requires careful lifetime management to prevent use-after-free while avoiding
//! runtime overhead.
//!
//! # Design
//!
//! The phantom lifetime approach encodes reference validity at the type level:
//!
//! ```text
//! ZeroCopyRef<'buffer, 'paradigm, T>
//!            │       │
//!            │       └─ Paradigm scope (prevents escaping)
//!            └─ Buffer lifetime (data validity)
//! ```
//!
//! # Safety Properties
//!
//! 1. References cannot outlive the underlying buffer
//! 2. References cannot escape their paradigm scope
//! 3. Mutable and immutable references are exclusive
//!
//! # Formal Model
//!
//! We model lifetimes as a partial order (L, ≤) where 'a ≤ 'b means 'a
//! lives at least as long as 'b. The type system ensures:
//!
//! ∀ ref: ZeroCopyRef<'buf, 'p, T>. 'buf ≤ 'ref ∧ 'p ≤ 'ref

use core::marker::PhantomData;
use core::ptr::NonNull;
use core::ops::{Deref, DerefMut};

/// A phantom lifetime marker for compile-time safety.
///
/// `PhantomLifetime` allows us to track lifetimes at the type level without
/// carrying any runtime data. This enables zero-cost safety guarantees.
///
/// # Example
///
/// ```rust
/// use nexus_memory::zero_copy::PhantomLifetime;
///
/// fn with_scope<'a, F, R>(f: F) -> R
/// where
///     F: FnOnce(PhantomLifetime<'a>) -> R,
/// {
///     f(PhantomLifetime::new())
/// }
/// ```
#[derive(Debug)]
pub struct PhantomLifetime<'a> {
    _marker: PhantomData<&'a ()>,
}

impl<'a> PhantomLifetime<'a> {
    /// Creates a new phantom lifetime marker.
    #[inline]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Converts to a shorter lifetime.
    ///
    /// This is safe because a reference valid for 'a is also valid for
    /// any shorter lifetime 'b where 'a: 'b.
    #[inline]
    pub fn shorten<'b>(self) -> PhantomLifetime<'b>
    where
        'a: 'b,
    {
        PhantomLifetime::new()
    }

    /// Intersects two lifetimes, returning the shorter one.
    #[inline]
    pub fn intersect<'b>(self, _other: PhantomLifetime<'b>) -> PhantomLifetime<'a>
    where
        'a: 'b,
    {
        self
    }
}

impl<'a> Clone for PhantomLifetime<'a> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<'a> Copy for PhantomLifetime<'a> {}

impl<'a> Default for PhantomLifetime<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// A zero-copy immutable reference with phantom lifetime tracking.
///
/// `ZeroCopyRef` wraps a pointer with two lifetime parameters:
/// - `'buf`: The lifetime of the underlying buffer
/// - `'scope`: The scope lifetime (typically a paradigm boundary)
///
/// The reference is only valid while both lifetimes are active.
///
/// # Example
///
/// ```rust
/// use nexus_memory::zero_copy::{ZeroCopyRef, PhantomLifetime};
///
/// fn process_batch<'a>(data: ZeroCopyRef<'a, 'a, [i32]>) {
///     // Can read data within this scope
///     let sum: i32 = data.iter().sum();
///     println!("Sum: {}", sum);
/// }
/// ```
pub struct ZeroCopyRef<'buf, 'scope, T: ?Sized> {
    /// Pointer to the data
    ptr: NonNull<T>,
    
    /// Buffer lifetime marker
    _buffer: PhantomData<&'buf T>,
    
    /// Scope lifetime marker
    _scope: PhantomData<&'scope ()>,
}

impl<'buf, 'scope, T: ?Sized> ZeroCopyRef<'buf, 'scope, T> {
    /// Creates a new zero-copy reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid, initialized memory
    /// - The data lives at least as long as `'buf`
    /// - The reference will not be used after `'scope` ends
    #[inline]
    pub const unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Creates a reference from a regular Rust reference.
    #[inline]
    pub fn from_ref(r: &'buf T) -> Self
    where
        'buf: 'scope,
    {
        Self {
            ptr: NonNull::from(r),
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Returns the underlying pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Converts to a regular reference.
    ///
    /// The returned reference is bounded by the shorter of the two lifetimes.
    #[inline]
    pub fn as_ref(&self) -> &'scope T
    where
        'buf: 'scope,
    {
        // SAFETY: Lifetime bounds ensure validity
        unsafe { self.ptr.as_ref() }
    }

    /// Shortens the scope lifetime.
    #[inline]
    pub fn narrow_scope<'shorter>(self) -> ZeroCopyRef<'buf, 'shorter, T>
    where
        'scope: 'shorter,
    {
        ZeroCopyRef {
            ptr: self.ptr,
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }
}

impl<'buf, 'scope, T: ?Sized> Clone for ZeroCopyRef<'buf, 'scope, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }
}

impl<'buf, 'scope, T: ?Sized> Copy for ZeroCopyRef<'buf, 'scope, T> {}

impl<'buf, 'scope, T: ?Sized> Deref for ZeroCopyRef<'buf, 'scope, T>
where
    'buf: 'scope,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

// SAFETY: ZeroCopyRef is essentially a reference
unsafe impl<'buf, 'scope, T: Sync + ?Sized> Send for ZeroCopyRef<'buf, 'scope, T> {}
unsafe impl<'buf, 'scope, T: Sync + ?Sized> Sync for ZeroCopyRef<'buf, 'scope, T> {}

/// A zero-copy mutable reference with phantom lifetime tracking.
///
/// Similar to `ZeroCopyRef`, but provides mutable access. Only one
/// `ZeroCopyMut` can exist for any given data at a time.
///
/// # Example
///
/// ```rust
/// use nexus_memory::zero_copy::{ZeroCopyMut, PhantomLifetime};
///
/// fn modify_data<'a>(mut data: ZeroCopyMut<'a, 'a, [i32]>) {
///     for x in data.iter_mut() {
///         *x *= 2;
///     }
/// }
/// ```
pub struct ZeroCopyMut<'buf, 'scope, T: ?Sized> {
    /// Pointer to the data
    ptr: NonNull<T>,
    
    /// Buffer lifetime marker
    _buffer: PhantomData<&'buf mut T>,
    
    /// Scope lifetime marker  
    _scope: PhantomData<&'scope mut ()>,
}

impl<'buf, 'scope, T: ?Sized> ZeroCopyMut<'buf, 'scope, T> {
    /// Creates a new mutable zero-copy reference.
    ///
    /// # Safety
    ///
    /// Same requirements as `ZeroCopyRef::new`, plus:
    /// - No other references to this data may exist
    #[inline]
    pub unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Creates a mutable reference from a regular Rust mutable reference.
    #[inline]
    pub fn from_mut(r: &'buf mut T) -> Self
    where
        'buf: 'scope,
    {
        Self {
            ptr: NonNull::from(r),
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Returns the underlying pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Converts to a regular mutable reference.
    #[inline]
    pub fn as_mut(&mut self) -> &'scope mut T
    where
        'buf: 'scope,
    {
        // SAFETY: We have exclusive access
        unsafe { self.ptr.as_mut() }
    }

    /// Downgrades to an immutable reference.
    #[inline]
    pub fn downgrade(self) -> ZeroCopyRef<'buf, 'scope, T> {
        ZeroCopyRef {
            ptr: self.ptr,
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Shortens the scope lifetime.
    #[inline]
    pub fn narrow_scope<'shorter>(self) -> ZeroCopyMut<'buf, 'shorter, T>
    where
        'scope: 'shorter,
    {
        ZeroCopyMut {
            ptr: self.ptr,
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }
}

impl<'buf, 'scope, T: ?Sized> Deref for ZeroCopyMut<'buf, 'scope, T>
where
    'buf: 'scope,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: Lifetime bounds ensure validity
        unsafe { self.ptr.as_ref() }
    }
}

impl<'buf, 'scope, T: ?Sized> DerefMut for ZeroCopyMut<'buf, 'scope, T>
where
    'buf: 'scope,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: We have exclusive access
        unsafe { self.ptr.as_mut() }
    }
}

// SAFETY: ZeroCopyMut provides exclusive access
unsafe impl<'buf, 'scope, T: Send + ?Sized> Send for ZeroCopyMut<'buf, 'scope, T> {}
unsafe impl<'buf, 'scope, T: Sync + ?Sized> Sync for ZeroCopyMut<'buf, 'scope, T> {}

/// A scoped region for zero-copy operations.
///
/// `ScopedRegion` provides a way to ensure that all zero-copy references
/// created within a scope are dropped before the scope exits.
///
/// # Example
///
/// ```rust
/// use nexus_memory::zero_copy::ScopedRegion;
///
/// fn example<T>(data: &[T]) {
///     ScopedRegion::new(|scope| {
///         // All references created here will be dropped when the closure returns
///     });
/// }
/// ```
pub struct ScopedRegion<'a> {
    _marker: PhantomData<&'a ()>,
}

impl<'a> ScopedRegion<'a> {
    /// Creates a new scoped region and invokes the callback.
    ///
    /// The callback receives a phantom lifetime marker that bounds any
    /// references created within the scope.
    #[inline]
    pub fn new<F, R>(f: F) -> R
    where
        F: for<'scope> FnOnce(PhantomLifetime<'scope>) -> R,
    {
        f(PhantomLifetime::new())
    }

    /// Creates a reference within this scope.
    #[inline]
    pub fn make_ref<T>(&self, data: &'a T) -> ZeroCopyRef<'a, 'a, T> {
        ZeroCopyRef::from_ref(data)
    }

    /// Creates a mutable reference within this scope.
    #[inline]
    pub fn make_mut<T>(&self, data: &'a mut T) -> ZeroCopyMut<'a, 'a, T> {
        ZeroCopyMut::from_mut(data)
    }
}

/// Lifetime arithmetic utilities
pub mod lifetime_arithmetic {
    use super::*;

    /// Computes the intersection of two lifetimes at the type level.
    ///
    /// Returns a marker for the shorter of the two lifetimes.
    #[inline]
    pub fn intersect<'a, 'b>() -> PhantomLifetime<'a>
    where
        'a: 'b,
    {
        PhantomLifetime::new()
    }

    /// Extends a reference's scope to a potentially longer lifetime.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the data actually lives for the
    /// extended lifetime.
    #[inline]
    pub unsafe fn extend_scope<'buf, 'short, 'long, T: ?Sized>(
        r: ZeroCopyRef<'buf, 'short, T>,
    ) -> ZeroCopyRef<'buf, 'long, T>
    where
        'long: 'short,
    {
        ZeroCopyRef {
            ptr: r.ptr,
            _buffer: PhantomData,
            _scope: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phantom_lifetime() {
        let _lt = PhantomLifetime::<'static>::new();
    }

    #[test]
    fn test_zero_copy_ref() {
        let data = 42i32;
        let r = ZeroCopyRef::from_ref(&data);
        
        assert_eq!(*r, 42);
    }

    #[test]
    fn test_zero_copy_mut() {
        let mut data = 42i32;
        let mut r = ZeroCopyMut::from_mut(&mut data);
        
        *r = 100;
        
        assert_eq!(data, 100);
    }

    #[test]
    fn test_downgrade() {
        let mut data = 42i32;
        let r = ZeroCopyMut::from_mut(&mut data);
        
        let immut = r.downgrade();
        assert_eq!(*immut, 42);
    }

    #[test]
    fn test_scoped_region() {
        let data = vec![1, 2, 3, 4, 5];
        
        ScopedRegion::new(|_scope| {
            let sum: i32 = data.iter().sum();
            assert_eq!(sum, 15);
        });
    }

    #[test]
    fn test_slice_ref() {
        let data = [1, 2, 3, 4, 5];
        let r: ZeroCopyRef<'_, '_, [i32]> = ZeroCopyRef::from_ref(&data[..]);
        
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], 1);
    }
}
