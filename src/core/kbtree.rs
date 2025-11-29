//! High-performance B-tree implementation for seed chaining.
//!
//! This is a Rust port of kbtree.h from klib, optimized for the specific
//! access patterns used in BWA-MEM2's seed chaining algorithm.
//!
//! Key design decisions:
//! - Generic over key type K (must be Ord + Copy)
//! - Values stored inline in nodes for cache efficiency
//! - Unsafe internals with safe public API
//! - Optimized for interval queries (find closest key)
//!
//! # Performance Characteristics
//! - O(log n) insert, lookup, interval query
//! - Cache-friendly node layout (keys contiguous in memory)
//! - No per-element heap allocation (bulk node allocation)
//!
//! # Usage
//! ```ignore
//! let mut tree: KBTree<u64, ChainData> = KBTree::new();
//! tree.insert(1000, chain_data);
//! if let Some((lower, upper)) = tree.interval(&500) {
//!     // lower = closest key <= 500
//!     // upper = closest key > 500
//! }
//! ```

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ptr;

/// Default node size in bytes (matches C++ KB_DEFAULT_SIZE)
pub const KB_DEFAULT_SIZE: usize = 512;

/// Minimum branching factor (t >= 2 for valid B-tree)
const MIN_T: usize = 2;

/// B-tree node header
#[repr(C)]
struct KBNodeHeader {
    /// Bit 0: is_internal, Bits 1-31: number of keys
    flags_and_n: u32,
}

impl KBNodeHeader {
    #[inline(always)]
    fn is_internal(&self) -> bool {
        (self.flags_and_n & 1) != 0
    }

    #[inline(always)]
    fn set_internal(&mut self, internal: bool) {
        if internal {
            self.flags_and_n |= 1;
        } else {
            self.flags_and_n &= !1;
        }
    }

    #[inline(always)]
    fn n(&self) -> usize {
        (self.flags_and_n >> 1) as usize
    }

    #[inline(always)]
    fn set_n(&mut self, n: usize) {
        self.flags_and_n = (self.flags_and_n & 1) | ((n as u32) << 1);
    }

    #[inline(always)]
    fn inc_n(&mut self) {
        self.flags_and_n += 2; // Add 2 because n is stored in bits 1-31
    }
}

/// High-performance B-tree optimized for interval queries.
///
/// Generic over:
/// - K: Key type (must be Ord + Copy + Default)
/// - V: Value type (must be Copy + Default)
pub struct KBTree<K, V>
where
    K: Ord + Copy + Default,
    V: Copy + Default,
{
    root: *mut u8,
    /// Offset to keys array within node
    off_key: usize,
    /// Offset to child pointers within node
    off_ptr: usize,
    /// Size of internal node in bytes
    ilen: usize,
    /// Size of leaf node in bytes
    elen: usize,
    /// Maximum keys per node (2t - 1)
    n: usize,
    /// Minimum degree
    t: usize,
    /// Total number of keys in tree
    n_keys: usize,
    /// Total number of nodes allocated
    n_nodes: usize,
    _phantom: PhantomData<(K, V)>,
}

impl<K, V> KBTree<K, V>
where
    K: Ord + Copy + Default,
    V: Copy + Default,
{
    /// Create a new B-tree with default node size.
    pub fn new() -> Self {
        Self::with_size(KB_DEFAULT_SIZE)
    }

    /// Create a new B-tree with specified node size in bytes.
    pub fn with_size(size: usize) -> Self {
        // Calculate branching factor t based on node size
        // Node layout: [header(4)] [padding] [keys: (K,V) * n] [ptrs: *mut u8 * (n+1)]
        let entry_size = std::mem::size_of::<(K, V)>();
        let entry_align = std::mem::align_of::<(K, V)>();
        let ptr_size = std::mem::size_of::<*mut u8>();
        let ptr_align = std::mem::align_of::<*mut u8>();
        let header_size = 4usize;

        // Keys must be aligned to entry_align
        let off_key = (header_size + entry_align - 1) & !(entry_align - 1);

        // Solve for max n given node size
        // off_key + n*entry + padding + (n+1)*ptr <= size
        // Approximate: max_n = (size - off_key - ptr) / (entry + ptr)
        let usable = size.saturating_sub(off_key + ptr_size);
        let max_n = usable / (entry_size + ptr_size);
        let t = max_n.div_ceil(2).max(MIN_T);
        let n = 2 * t - 1;

        // Calculate pointer offset (after keys array, aligned)
        let keys_end = off_key + n * entry_size;
        let off_ptr = (keys_end + ptr_align - 1) & !(ptr_align - 1);

        // Internal node size (with child pointers), 8-byte aligned
        let ilen = (off_ptr + (n + 1) * ptr_size + 7) & !7;
        // Leaf node size (no child pointers), 8-byte aligned
        let elen = (keys_end + 7) & !7;

        // Allocate root node (starts as leaf)
        let root = unsafe {
            let layout = Layout::from_size_align(ilen, 8).unwrap();
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                panic!("KBTree: failed to allocate root node");
            }
            ptr
        };

        KBTree {
            root,
            off_key,
            off_ptr,
            ilen,
            elen,
            n,
            t,
            n_keys: 0,
            n_nodes: 1,
            _phantom: PhantomData,
        }
    }

    /// Returns the number of keys in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_keys
    }

    /// Returns true if the tree is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_keys == 0
    }

    /// Get pointer to keys array in node
    #[inline(always)]
    unsafe fn keys(&self, node: *mut u8) -> *mut (K, V) {
        unsafe { node.add(self.off_key) as *mut (K, V) }
    }

    /// Get pointer to child pointers array in node
    #[inline(always)]
    unsafe fn children(&self, node: *mut u8) -> *mut *mut u8 {
        unsafe { node.add(self.off_ptr) as *mut *mut u8 }
    }

    /// Get node header
    #[inline(always)]
    unsafe fn header(&self, node: *mut u8) -> *mut KBNodeHeader {
        node as *mut KBNodeHeader
    }

    /// Binary search for key position in node.
    /// Returns (index, found) where:
    /// - If found: keys[index] == key
    /// - If not found: keys[index] is largest key < key (or -1 if key < all keys)
    unsafe fn search_node(&mut self, node: *mut u8, key: &K) -> (i32, bool) {
        let header_ptr = unsafe { self.header(node) };
        let header = unsafe { &mut *header_ptr };
        let n = header.n();
        if n == 0 {
            return (-1, false);
        }

        let keys = unsafe { self.keys(node) };
        let mut begin = 0i32;
        let mut end = n as i32;

        // Binary search
        while begin < end {
            let mid = (begin + end) / 2;
            match unsafe { (*keys.offset(mid as isize)).0.cmp(key) } {
                Ordering::Less => begin = mid + 1,
                Ordering::Greater => end = mid,
                Ordering::Equal => return (mid, true),
            }
        }

        // begin is insertion point; return begin-1 as the largest key < key
        if begin == 0 {
            (-1, false)
        } else {
            let idx = begin - 1;
            let found = unsafe { (*keys.offset(idx as isize)).0.cmp(key) } == Ordering::Equal;
            (idx, found)
        }
    }

    /// Find interval containing key.
    /// Returns (lower, upper) where:
    /// - lower: reference to entry with largest key <= query (None if query < all keys)
    /// - upper: reference to entry with smallest key > query (None if query >= all keys)
    pub fn interval(&mut self, key: &K) -> (Option<&(K, V)>, Option<&(K, V)>) {
        if self.n_keys == 0 {
            return (None, None);
        }

        let mut lower: *const (K, V) = ptr::null();
        let mut upper: *const (K, V) = ptr::null();
        let mut x = self.root;

        while !x.is_null() {
            let header_ptr = unsafe { self.header(x) };
            let header = unsafe { &mut *header_ptr };
            let n = header.n();
            let keys = unsafe { self.keys(x) };

            let (i, found) = unsafe { self.search_node(x, key) };

            if found {
                // Exact match
                let entry = unsafe { keys.offset(i as isize) };
                return (Some(unsafe { &*entry }), Some(unsafe { &*entry }));
            }

            // Update lower bound (largest key <= query)
            if i >= 0 {
                lower = unsafe { keys.offset(i as isize) };
            }

            // Update upper bound (smallest key > query)
            let next_idx = i + 1;
            if next_idx < n as i32 {
                upper = unsafe { keys.offset(next_idx as isize) };
            }

            // Descend to child
            if !header.is_internal() {
                break;
            }
            x = unsafe { *self.children(x).offset(next_idx as isize) };
        }

        let lower_ref = if lower.is_null() {
            None
        } else {
            Some(unsafe { &*lower })
        };
        let upper_ref = if upper.is_null() {
            None
        } else {
            Some(unsafe { &*upper })
        };

        (lower_ref, upper_ref)
    }

    /// Get mutable reference to value with largest key <= query.
    /// This is the primary operation used in seed chaining.
    pub fn get_lower_mut(&mut self, key: &K) -> Option<&mut (K, V)> {
        if self.n_keys == 0 {
            return None;
        }

        let mut lower: *mut (K, V) = ptr::null_mut();
        let mut x = self.root;

        while !x.is_null() {
            let header_ptr = unsafe { self.header(x) };
            let header = unsafe { &mut *header_ptr };
            let keys = unsafe { self.keys(x) };

            let (i, found) = unsafe { self.search_node(x, key) };

            if found {
                return Some(unsafe { &mut *keys.offset(i as isize) });
            }

            if i >= 0 {
                lower = unsafe { keys.offset(i as isize) };
            }

            if !header.is_internal() {
                break;
            }
            x = unsafe { *self.children(x).offset((i + 1) as isize) };
        }

        if lower.is_null() {
            None
        } else {
            Some(unsafe { &mut *lower })
        }
    }

    /// Allocate a new node
    unsafe fn alloc_node(&mut self, internal: bool) -> *mut u8 {
        let size = if internal { self.ilen } else { self.elen };
        let layout = Layout::from_size_align(size, 8).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            panic!("KBTree: failed to allocate node");
        }
        self.n_nodes += 1;

        let header_ptr = unsafe { self.header(ptr) };
        let header = unsafe { &mut *header_ptr };
        header.set_internal(internal);
        header.set_n(0);

        ptr
    }

    /// Split child y of node x at index i
    unsafe fn split_child(&mut self, x: *mut u8, i: usize, y: *mut u8) {
        let header_y_ptr = unsafe { self.header(y) };
        let header_y = unsafe { &mut *header_y_ptr };
        let y_internal = header_y.is_internal();

        // Allocate new node z
        let z = unsafe { self.alloc_node(y_internal) };

        let header_z_ptr = unsafe { self.header(z) };
        let header_z = unsafe { &mut *header_z_ptr };
        header_z.set_n(self.t - 1);

        // Copy upper half of y's keys to z
        let keys_y = unsafe { self.keys(y) };
        let keys_z = unsafe { self.keys(z) };
        unsafe { ptr::copy_nonoverlapping(keys_y.add(self.t), keys_z, self.t - 1) };

        // Copy upper half of y's children to z (if internal)
        if y_internal {
            let children_y = unsafe { self.children(y) };
            let children_z = unsafe { self.children(z) };
            unsafe { ptr::copy_nonoverlapping(children_y.add(self.t), children_z, self.t) };
        }

        // Reduce y's key count
        let header_y_ptr = unsafe { self.header(y) };
        let header_y = unsafe { &mut *header_y_ptr };
        header_y.set_n(self.t - 1);

        // Shift x's children to make room for z
        let children_x = unsafe { self.children(x) };
        let header_x_ptr = unsafe { self.header(x) };
        let header_x = unsafe { &mut *header_x_ptr };
        let n_x = header_x.n();

        for j in (i + 1..=n_x).rev() {
            unsafe { *children_x.add(j + 1) = *children_x.add(j) };
        }
        unsafe { *children_x.add(i + 1) = z };

        // Shift x's keys and insert median from y
        let keys_x = unsafe { self.keys(x) };
        for j in (i..n_x).rev() {
            unsafe { *keys_x.add(j + 1) = *keys_x.add(j) };
        }
        unsafe { *keys_x.add(i) = *keys_y.add(self.t - 1) };

        header_x.inc_n();
    }

    /// Insert into non-full node
    unsafe fn insert_nonfull(&mut self, x: *mut u8, key: K, value: V) {
        let header_ptr = unsafe { self.header(x) };
        let header = unsafe { &mut *header_ptr };
        let mut i = header.n() as i32 - 1;
        let keys = unsafe { self.keys(x) };

        if !header.is_internal() {
            // Leaf node: shift keys and insert
            while i >= 0 && unsafe { (*keys.offset(i as isize)).0 } > key {
                unsafe { *keys.offset((i + 1) as isize) = *keys.offset(i as isize) };
                i -= 1;
            }
            unsafe { *keys.offset((i + 1) as isize) = (key, value) };
            header.inc_n();
        } else {
            // Internal node: find child and recurse
            while i >= 0 && unsafe { (*keys.offset(i as isize)).0 } > key {
                i -= 1;
            }
            i += 1;

            let children = unsafe { self.children(x) };
            let child = unsafe { *children.offset(i as isize) };
            let child_header_ptr = unsafe { self.header(child) };
            let child_header = unsafe { &mut *child_header_ptr };

            if child_header.n() == self.n {
                // Child is full, split it
                unsafe { self.split_child(x, i as usize, child) };
                if unsafe { (*keys.offset(i as isize)).0 } < key {
                    i += 1;
                }
            }

            unsafe { self.insert_nonfull(*children.offset(i as isize), key, value) };
        }
    }

    /// Insert a key-value pair into the tree.
    pub fn insert(&mut self, key: K, value: V) {
        self.n_keys += 1;

        unsafe {
            let root_header_ptr = self.header(self.root);
            let root_header = &mut *root_header_ptr;
            if root_header.n() == self.n {
                // Root is full, create new root
                let new_root = self.alloc_node(true);
                let children = self.children(new_root);
                *children = self.root;

                self.split_child(new_root, 0, self.root);
                self.root = new_root;

                self.insert_nonfull(new_root, key, value);
            } else {
                self.insert_nonfull(self.root, key, value);
            }
        }
    }

    /// Iterate over all entries in sorted order.
    /// Callback receives (&K, &V) for each entry.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V),
    {
        if self.n_keys == 0 {
            return;
        }

        unsafe { self.traverse_node(self.root, &mut f) };
    }

    /// Traverse a node and its children in-order
    unsafe fn traverse_node<F>(&self, node: *mut u8, f: &mut F)
    where
        F: FnMut(&K, &V),
    {
        let header_ptr = unsafe { self.header(node) };
        let header = unsafe { &mut *header_ptr };
        let n = header.n();
        let keys = unsafe { self.keys(node) };

        if header.is_internal() {
            let children = unsafe { self.children(node) };
            for i in 0..n {
                unsafe { self.traverse_node(*children.add(i), f) };
                let entry = unsafe { &*keys.add(i) };
                f(&entry.0, &entry.1);
            }
            unsafe { self.traverse_node(*children.add(n), f) };
        } else {
            for i in 0..n {
                let entry = unsafe { &*keys.add(i) };
                f(&entry.0, &entry.1);
            }
        }
    }

    /// Collect all entries into a Vec in sorted order.
    pub fn to_vec(&self) -> Vec<(K, V)> {
        let mut result = Vec::with_capacity(self.n_keys);
        self.for_each(|k, v| result.push((*k, *v)));
        result
    }

    /// Free a node and all its children recursively
    unsafe fn free_node(&mut self, node: *mut u8) {
        if node.is_null() {
            return;
        }

        let header_ptr = unsafe { self.header(node) };
        let header = unsafe { &mut *header_ptr };
        let is_internal = header.is_internal();
        let n = header.n();

        if is_internal {
            let children = unsafe { self.children(node) };
            for i in 0..=n {
                let child = unsafe { *children.add(i) };
                if !child.is_null() {
                    unsafe { self.free_node(child) };
                }
            }
        }

        let size = if is_internal { self.ilen } else { self.elen };
        let layout = Layout::from_size_align(size, 8).unwrap();
        unsafe { dealloc(node, layout) };
        self.n_nodes -= 1;
    }
}

impl<K, V> Default for KBTree<K, V>
where
    K: Ord + Copy + Default,
    V: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for KBTree<K, V>
where
    K: Ord + Copy + Default,
    V: Copy + Default,
{
    fn drop(&mut self) {
        unsafe {
            self.free_node(self.root);
        }
    }
}

// Safety: KBTree can be sent between threads if K and V are Send
unsafe impl<K, V> Send for KBTree<K, V>
where
    K: Ord + Copy + Default + Send,
    V: Copy + Default + Send,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let mut tree: KBTree<u64, i32> = KBTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.interval(&100).0.is_none());
        assert!(tree.interval(&100).1.is_none());
    }

    #[test]
    fn test_single_insert() {
        let mut tree: KBTree<u64, i32> = KBTree::new();
        tree.insert(100, 42);
        assert_eq!(tree.len(), 1);

        let (lower, upper) = tree.interval(&100);
        assert_eq!(lower.map(|e| e.0), Some(100));
        assert_eq!(upper.map(|e| e.0), Some(100));
    }

    #[test]
    fn test_interval_query() {
        let mut tree: KBTree<u64, i32> = KBTree::new();
        tree.insert(100, 1);
        tree.insert(200, 2);
        tree.insert(300, 3);

        // Query below all keys
        let (lower, upper) = tree.interval(&50);
        assert!(lower.is_none());
        assert_eq!(upper.map(|e| e.0), Some(100));

        // Query between keys
        let (lower, upper) = tree.interval(&150);
        assert_eq!(lower.map(|e| e.0), Some(100));
        assert_eq!(upper.map(|e| e.0), Some(200));

        // Query above all keys
        let (lower, upper) = tree.interval(&350);
        assert_eq!(lower.map(|e| e.0), Some(300));
        assert!(upper.is_none());
    }

    #[test]
    fn test_many_inserts() {
        let mut tree: KBTree<u64, i32> = KBTree::new();

        // Insert 1000 elements in random order
        let values: Vec<u64> = (0..1000).map(|i| (i * 7919) % 10000).collect();
        for (i, &v) in values.iter().enumerate() {
            tree.insert(v, i as i32);
        }

        assert_eq!(tree.len(), 1000);

        // Verify traversal is sorted
        let result = tree.to_vec();
        for i in 1..result.len() {
            assert!(result[i - 1].0 <= result[i].0);
        }
    }

    #[test]
    fn test_get_lower_mut() {
        let mut tree: KBTree<u64, i32> = KBTree::new();
        tree.insert(100, 1);
        tree.insert(200, 2);
        tree.insert(300, 3);

        // Modify value through get_lower_mut
        if let Some(entry) = tree.get_lower_mut(&150) {
            entry.1 = 999;
        }

        let (lower, _) = tree.interval(&150);
        assert_eq!(lower.map(|e| e.1), Some(999));
    }

    #[test]
    fn test_node_splits() {
        let mut tree: KBTree<u64, i32> = KBTree::with_size(64); // Small nodes to force splits

        // Insert enough to trigger multiple splits
        for i in 0..100 {
            tree.insert(i, i as i32);
        }

        assert_eq!(tree.len(), 100);

        // Verify all elements present
        let result = tree.to_vec();
        assert_eq!(result.len(), 100);
        for i in 0u64..100 {
            assert_eq!(result[i as usize], (i, i as i32));
        }
    }

    #[test]
    fn test_descending_inserts() {
        let mut tree: KBTree<u64, i32> = KBTree::new();

        // Insert in descending order (worst case for naive implementations)
        for i in (0..500).rev() {
            tree.insert(i, i as i32);
        }

        assert_eq!(tree.len(), 500);

        let result = tree.to_vec();
        for i in 0u64..500 {
            assert_eq!(result[i as usize], (i, i as i32));
        }
    }
}
