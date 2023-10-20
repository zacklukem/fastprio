//! A fast priority queue implementation using a fixed number of buckets.
//!
//! The generic queue is implemented as [`FastPriorityQueueImpl`].  For a type
//! using `VecDeque` as the underlying queue, use [`FastPriorityQueue`].
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs, unused_must_use, unsafe_code)]

#[cfg(feature = "std")]
mod stdimpl {
    use super::*;
    use std::collections::VecDeque;

    /// A fast priority queue implementation using VecDeque with a fixed number of buckets.
    pub type FastPriorityQueue<T, const DOMAIN: usize> = FastPriorityQueueImpl<VecDeque<T>, DOMAIN>;

    impl<T> Queue for VecDeque<T> {
        type Item = T;
        fn front(&self) -> Option<&T> {
            self.front()
        }

        fn push_back(&mut self, item: T) {
            self.push_back(item)
        }

        fn pop_front(&mut self) -> Option<T> {
            self.pop_front()
        }

        fn len(&self) -> usize {
            self.len()
        }
    }

    impl<T, const N: usize> FastPriorityQueueImpl<VecDeque<T>, N> {
        const EL: VecDeque<T> = VecDeque::new();
        /// Create a new empty `FastPriorityQueue`.
        ///
        /// # Example
        /// ```
        /// use fastprio::FastPriorityQueue;
        ///
        /// let mut queue = FastPriorityQueue::<_, 16>::new_const();
        /// queue.push(5, "foo".to_string()).unwrap();
        /// ```
        ///
        pub const fn new_const() -> Self {
            Self {
                buckets: [Self::EL; N],
                max_prio: 0,
            }
        }
    }
}

#[cfg(feature = "std")]
pub use stdimpl::FastPriorityQueue;

#[cfg(feature = "heapless")]
mod heaplessimpl {
    use super::*;
    use heapless::Deque;

    impl<T, const N: usize> Queue for Deque<T, N> {
        type Item = T;
        fn front(&self) -> Option<&T> {
            self.front()
        }

        fn push_back(&mut self, item: T) {
            self.push_back(item)
                .map_err(|_| "overflowed heapless deque")
                .unwrap()
        }

        fn pop_front(&mut self) -> Option<T> {
            self.pop_front()
        }

        fn len(&self) -> usize {
            self.len()
        }
    }

    impl<T, const N: usize, const N2: usize> FastPriorityQueueImpl<Deque<T, N2>, N> {
        const EL: Deque<T, N2> = Deque::new();
        /// Create a new empty `FastPriorityQueueImpl` with a heapless deque as
        /// the container.
        ///
        /// # Example
        /// ```
        /// use fastprio::FastPriorityQueueImpl;
        /// use heapless::Deque;
        ///
        /// let mut queue = FastPriorityQueueImpl::<Deque<_, 16>, 16>::new_const();
        /// queue.push(5, "foo".to_string()).unwrap();
        /// ```
        ///
        pub const fn new_const() -> Self {
            Self {
                buckets: [Self::EL; N],
                max_prio: 0,
            }
        }
    }
}

use core::fmt::Debug;

/// A trait for queue implementations. This is used to allow multiple queue
/// implementations to be used with the same priority queue.
///
/// This type is implemented for `VecDeque` by default.
pub trait Queue: Default {
    /// The type of item stored in the queue.
    type Item;

    /// Get the item at the front of the queue
    fn front(&self) -> Option<&Self::Item>;

    /// Push an item to the back of the queue
    fn push_back(&mut self, item: Self::Item);

    /// Pop an item from the front of the queue
    fn pop_front(&mut self) -> Option<Self::Item>;

    /// Get the number of items in the queue
    fn len(&self) -> usize;

    /// Check if the queue is empty
    fn is_empty(&self) -> bool {
        self.front().is_none()
    }
}

/// A fast priority queue implementation using a fixed number of buckets.
///
/// The advantage of this queue is that it is *O*(1) (with some caveats) for all
/// operations.
///
/// This queue is more limited than the standard library's `BinaryHeap` in that
/// the priority must be a `usize` and the domain of priorities is
/// `[0, DOMAIN)`.
///
/// # Examples
/// ```
/// use fastprio::FastPriorityQueue;
/// let mut queue = FastPriorityQueue::<_, 16>::from_iter([(0, "foo"), (1, "bar")]);
/// assert_eq!(queue.pop(), Some((1, "bar")));
/// queue.push(5, "baz").unwrap();
/// queue.push(5, "qux").unwrap();
/// assert_eq!(queue.peek(), Some((5, &"baz")));
/// ```
#[derive(Clone)]
pub struct FastPriorityQueueImpl<Container, const DOMAIN: usize> {
    buckets: [Container; DOMAIN],
    max_prio: usize,
}

impl<T, Container, const DOMAIN: usize> Debug for FastPriorityQueueImpl<Container, DOMAIN>
where
    T: Debug,
    Container: Queue<Item = T>,
    for<'a> &'a Container: IntoIterator<Item = &'a T>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, Container, const DOMAIN: usize> Default for FastPriorityQueueImpl<Container, DOMAIN>
where
    Container: Queue<Item = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Container, const DOMAIN: usize> FastPriorityQueueImpl<Container, DOMAIN>
where
    Container: Queue<Item = T>,
{
    /// Create a new empty [`FastPriorityQueueImpl`].
    ///
    /// For a const queue, use `new_const` instead.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// let mut queue = FastPriorityQueue::<_, 16>::new();
    /// queue.push(5, "foo".to_string()).unwrap();
    /// ```
    ///
    pub fn new() -> Self {
        Self {
            buckets: core::array::from_fn(|_| Container::default()),
            max_prio: 0,
        }
    }

    /// Create a new empty [`FastPriorityQueueImpl`] using the given bucket
    /// array.
    ///
    /// This is useful for creating a const queue with a custom `Container`
    /// type.  If you are using `VecDeque`, use [`FastPriorityQueue::new_const`]
    /// instead.
    ///
    /// The all the buckets must be empty.  If not, this will cause incorrect
    /// behavior.
    ///
    /// # Example
    /// ```
    /// use fastprio::{FastPriorityQueueImpl, Queue};
    ///
    /// #[derive(Default)]
    /// struct MyQueueType(/* fields */);
    /// impl Queue for MyQueueType {
    ///     type Item = u8;
    ///     
    ///     fn front(&self) -> Option<&Self::Item> { None }
    ///     fn push_back(&mut self, item: Self::Item) { todo!() }
    ///     fn pop_front(&mut self) -> Option<Self::Item> { None }
    ///     fn len(&self) -> usize { 0 }
    /// }
    ///
    /// const fn create_queue() -> FastPriorityQueueImpl<MyQueueType, 16> {
    ///     const VAL: MyQueueType = MyQueueType();
    ///     FastPriorityQueueImpl::from_buckets([VAL; 16])
    /// }
    ///
    /// static QUEUE: FastPriorityQueueImpl<MyQueueType, 16> = create_queue();
    /// assert_eq!(QUEUE.len(), 0);
    /// ```
    pub const fn from_buckets(buckets: [Container; DOMAIN]) -> Self {
        Self {
            buckets,
            max_prio: 0,
        }
    }

    /// Check if a priority is valid for this queue type.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// assert!(FastPriorityQueue::<u8, 16>::is_valid_prio(5));
    /// assert!(!FastPriorityQueue::<u8, 16>::is_valid_prio(16));
    /// assert!(!FastPriorityQueue::<u8, 16>::is_valid_prio(17));
    /// ```
    pub const fn is_valid_prio(prio: usize) -> bool {
        prio < DOMAIN
    }

    /// Push an item into the queue. Returns `Err` with the item and priority if
    /// the priority is invalid.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// let mut queue = FastPriorityQueue::<_, 16>::new();
    /// queue.push(4, "bar").unwrap();
    /// queue.push(5, "foo").unwrap();
    /// queue.push(4, "baz").unwrap();
    /// assert_eq!(queue.len(), 3);
    /// assert_eq!(queue.peek(), Some((5, &"foo")));
    /// ```
    ///
    /// # Time complexity
    /// This push operation is always *O*(1) when `Container::push_back` is
    /// *O*(1). This is usually the case when using the standard library's
    /// `VecDeque`. (Sometimes reallocation is *O*(n) for `VecDeque`)
    pub fn push(&mut self, prio: usize, item: T) -> Result<(), (usize, T)> {
        if !Self::is_valid_prio(prio) {
            return Err((prio, item));
        }
        self.buckets[prio].push_back(item);
        self.max_prio = self.max_prio.max(prio);
        Ok(())
    }

    /// Get the number of items in the queue.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// let queue = FastPriorityQueue::<_, 16>::from_iter([(0, "foo"), (1, "bar")]);
    /// assert_eq!(queue.len(), 2);
    /// ```
    ///
    /// # Time complexity
    /// The len operation is *O*(`DOMAIN`). This can generally be considered
    /// *O*(1), however for large values of DOMAIN, this can be a significant
    /// cost.
    pub fn len(&self) -> usize {
        self.buckets[0..=self.max_prio]
            .iter()
            .map(|bucket| bucket.len())
            .sum()
    }

    /// Check if the queue is empty.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// let mut queue = FastPriorityQueue::<_, 16>::new();
    /// assert!(queue.is_empty());
    /// queue.push(5, "foo").unwrap();
    /// assert!(!queue.is_empty());
    /// ```
    ///
    /// # Time complexity
    /// This push operation is always *O*(1) when `Container::front` is
    /// *O*(1). This is always the case when using the standard library's
    /// `VecDeque`.
    pub fn is_empty(&self) -> bool {
        self.peek().is_none()
    }

    /// Get the item with the highest priority.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// let mut queue = FastPriorityQueue::<_, 16>::from_iter([(4, "bar"), (5, "foo"), (4, "baz")]);
    /// assert_eq!(queue.pop(), Some((5, "foo")));
    /// ```
    ///
    /// # Time complexity
    /// The len operation is *O*(`DOMAIN`). This can generally be considered
    /// *O*(1), however for large values of DOMAIN, this can be a significant
    /// cost.
    pub fn pop(&mut self) -> Option<(usize, T)> {
        let bucket = &mut self.buckets[self.max_prio];

        let out = bucket.pop_front().map(|item| (self.max_prio, item));

        if bucket.is_empty() {
            self.max_prio = self.buckets[0..self.max_prio]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, bucket)| !bucket.is_empty())
                .map(|(prio, _)| prio)
                .unwrap_or(0);
        }

        out
    }

    /// Get the item with the highest priority without removing it from the queue.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// let queue = FastPriorityQueue::<_, 16>::from_iter([(0, "foo"), (1, "bar")]);
    /// assert_eq!(queue.peek(), Some((1, &"bar")));
    /// ```
    ///
    /// # Time complexity
    /// This push operation is always *O*(1) when `Container::front` is
    /// *O*(1). This is always the case when using the standard library's
    /// `VecDeque`.
    pub fn peek(&self) -> Option<(usize, &T)> {
        self.buckets[self.max_prio]
            .front()
            .map(|item| (self.max_prio, item))
    }
}

impl<T, Container, const DOMAIN: usize> FromIterator<(usize, T)>
    for FastPriorityQueueImpl<Container, DOMAIN>
where
    Container: Queue<Item = T>,
{
    fn from_iter<I: IntoIterator<Item = (usize, T)>>(iter: I) -> Self {
        let mut out = Self::new();
        for (prio, item) in iter {
            out.push(prio, item)
                .map_err(|_| "Invalid priority when calling FastPriorityQueueImpl::from_iter")
                .unwrap();
        }
        out
    }
}

impl<T, Container, const DOMAIN: usize> FastPriorityQueueImpl<Container, DOMAIN>
where
    Container: Queue<Item = T>,
    for<'a> &'a Container: IntoIterator<Item = &'a T>,
{
    /// Get an iterator over the items in the queue in order.
    ///
    /// # Example
    /// ```
    /// use fastprio::FastPriorityQueue;
    /// let queue = FastPriorityQueue::<_, 16>::from_iter([(4, "bar"), (5, "foo"), (4, "baz")]);
    /// let mut it = queue.iter();
    /// assert_eq!(it.next(), Some((5, &"foo")));
    /// assert_eq!(it.next(), Some((4, &"bar")));
    /// assert_eq!(it.next(), Some((4, &"baz")));
    /// ```
    pub fn iter(&self) -> FastPriorityQueueIter<T, Container, DOMAIN> {
        FastPriorityQueueIter::<T, Container, DOMAIN>::new(self)
    }
}

impl<'a, T, Container, const DOMAIN: usize> IntoIterator
    for &'a FastPriorityQueueImpl<Container, DOMAIN>
where
    Container: Queue<Item = T>,
    &'a Container: IntoIterator<Item = &'a T>,
    T: 'a,
{
    type Item = (usize, &'a T);
    type IntoIter = FastPriorityQueueIter<'a, T, Container, DOMAIN>;

    fn into_iter(self) -> Self::IntoIter {
        FastPriorityQueueIter::<T, Container, DOMAIN>::new(self)
    }
}

impl<T, Container, const DOMAIN: usize> IntoIterator for FastPriorityQueueImpl<Container, DOMAIN>
where
    Container: Queue<Item = T>,
{
    type Item = (usize, T);
    type IntoIter = FastPriorityQueueIntoIter<Container, DOMAIN>;

    fn into_iter(self) -> Self::IntoIter {
        FastPriorityQueueIntoIter { queue: self }
    }
}

/// An iterator over the items in a [`FastPriorityQueueImpl`].
pub struct FastPriorityQueueIter<'a, T, Container, const DOMAIN: usize>
where
    &'a Container: IntoIterator<Item = &'a T>,
    T: 'a,
{
    queue: &'a FastPriorityQueueImpl<Container, DOMAIN>,
    current_bucket: <&'a Container as IntoIterator>::IntoIter,
    bucket_idx: usize,
}

impl<'a, T, Container, const DOMAIN: usize> FastPriorityQueueIter<'a, T, Container, DOMAIN>
where
    Container: Queue<Item = T>,
    &'a Container: IntoIterator<Item = &'a T>,
    T: 'a,
{
    fn new(queue: &'a FastPriorityQueueImpl<Container, DOMAIN>) -> Self {
        let current_bucket = queue.buckets[queue.max_prio].into_iter();
        Self {
            queue,
            current_bucket,
            bucket_idx: queue.max_prio,
        }
    }
}

impl<'a, T, Container, const DOMAIN: usize> Iterator
    for FastPriorityQueueIter<'a, T, Container, DOMAIN>
where
    Container: Queue<Item = T>,
    &'a Container: IntoIterator<Item = &'a T>,
    T: 'a,
{
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.current_bucket.next() {
            Some((self.bucket_idx, v))
        } else if self.bucket_idx == 0 {
            None
        } else {
            self.bucket_idx = self.queue.buckets[0..self.bucket_idx]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, bucket)| !bucket.is_empty())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            self.current_bucket = self.queue.buckets[self.bucket_idx].into_iter();
            self.current_bucket.next().map(|v| (self.bucket_idx, v))
        }
    }
}

/// An iterator over the items in a [`FastPriorityQueueImpl`] that drains the values.
pub struct FastPriorityQueueIntoIter<Container, const DOMAIN: usize> {
    queue: FastPriorityQueueImpl<Container, DOMAIN>,
}

impl<T, Container, const DOMAIN: usize> Iterator for FastPriorityQueueIntoIter<Container, DOMAIN>
where
    Container: Queue<Item = T>,
{
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop()
    }
}

impl<T, Container, const DOMAIN: usize> Extend<(usize, T)>
    for FastPriorityQueueImpl<Container, DOMAIN>
where
    Container: Queue<Item = T>,
{
    fn extend<I: IntoIterator<Item = (usize, T)>>(&mut self, iter: I) {
        for (prio, item) in iter {
            self.push(prio, item)
                .map_err(|_| "Invalid priority when calling FastPriorityQueueImpl::extend")
                .unwrap();
        }
    }
}
