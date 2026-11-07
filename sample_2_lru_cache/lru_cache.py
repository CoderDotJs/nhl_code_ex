from __future__ import annotations

import time
from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

K = TypeVar("K")
V = TypeVar("V")


class Node(Generic[K, V]):
    """Internal doubly linked list node holding key, value, and expiration."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(
        self,
        key: K,
        value: V,
        expires_at: Optional[float] = None,
    ) -> None:
        self.key: K = key
        self.value: V = value
        self.expires_at: Optional[float] = expires_at
        self.prev: Optional[Node[K, V]] = None
        self.next: Optional[Node[K, V]] = None


class LRUCache(Generic[K, V]):
    """O(1) Least Recently Used cache using a Doubly Linked List and Hash Map.

    The head sentinel sits right before the most recently used (MRU) node,
    and the tail sentinel sits right after the least recently used (LRU) node.
    """

    def __init__(
        self,
        capacity: int,
        default_ttl: Optional[float] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Cache capacity must be >= 1, got {capacity}")
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError(f"default_ttl must be positive, got {default_ttl}")

        self._capacity: int = capacity
        self._default_ttl: Optional[float] = default_ttl
        self._lookup: Dict[K, Node[K, V]] = {}

        # Dummy sentinel boundaries to avoid edge-case None checks
        self._head: Node[K, V] = Node(key=None, value=None)  # type: ignore
        self._tail: Node[K, V] = Node(key=None, value=None)  # type: ignore
        self._head.next = self._tail
        self._tail.prev = self._head

    @property
    def capacity(self) -> int:
        """Maximum allowable entries."""
        return self._capacity

    def __len__(self) -> int:
        """Current number of active non-expired entries in cache."""
        return len(self._lookup)

    def __contains__(self, key: K) -> bool:
        """Checks if key exists and has not expired."""
        node = self._lookup.get(key)
        if node is None:
            return False
        if self._is_expired(node):
            self._remove_node(node)
            del self._lookup[key]
            return False
        return True

    def _is_expired(self, node: Node[K, V]) -> bool:
        """Checks if a node has exceeded its TTL timestamp."""
        if node.expires_at is None:
            return False
        return time.monotonic() >= node.expires_at

    def _detach_node(self, node: Node[K, V]) -> None:
        """Unlinks a node from its current position in the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node

    def _insert_after_head(self, node: Node[K, V]) -> None:
        """Places a node right after the head sentinel (marking it Most Recently Used)."""
        first_real = self._head.next
        node.prev = self._head
        node.next = first_real

        self._head.next = node
        if first_real is not None:
            first_real.prev = node

    def _move_to_front(self, node: Node[K, V]) -> None:
        """Moves an existing node to the head of the list (MRU)."""
        self._detach_node(node)
        self._insert_after_head(node)

    def _remove_node(self, node: Node[K, V]) -> None:
        """Completely detaches node from list."""
        self._detach_node(node)
        node.prev = None
        node.next = None

    def _pop_lru(self) -> Optional[Node[K, V]]:
        """Removes and returns the least recently used node right before tail sentinel."""
        lru_node = self._tail.prev
        if lru_node is None or lru_node is self._head:
            return None
        self._remove_node(lru_node)
        return lru_node

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Retrieves item value and refreshes its LRU position to most recent."""
        node = self._lookup.get(key)
        if node is None:
            return default

        # Expired entry check
        if self._is_expired(node):
            self._remove_node(node)
            del self._lookup[key]
            return default

        self._move_to_front(node)
        return node.value

    def peek(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Retrieves item value without updating its LRU priority."""
        node = self._lookup.get(key)
        if node is None:
            return default

        if self._is_expired(node):
            self._remove_node(node)
            del self._lookup[key]
            return default

        return node.value

    def put(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None,
    ) -> None:
        """Inserts or updates a key-value pair, refreshing its LRU priority."""
        lifespan = ttl if ttl is not None else self._default_ttl
        expires_at = (time.monotonic() + lifespan) if lifespan is not None else None

        node = self._lookup.get(key)
        if node is not None:
            # Key already exists: update payload and move to front
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
            return

        # New key: check capacity and evict LRU if needed
        if len(self._lookup) >= self._capacity:
            evicted = self._pop_lru()
            if evicted is not None and evicted.key in self._lookup:
                del self._lookup[evicted.key]

        new_node = Node(key=key, value=value, expires_at=expires_at)
        self._insert_after_head(new_node)
        self._lookup[key] = new_node

    def delete(self, key: K) -> bool:
        """Removes key from cache if present. Returns True if removed."""
        node = self._lookup.pop(key, None)
        if node is not None:
            self._remove_node(node)
            return True
        return False

    def clear(self) -> None:
        """Empties all elements from the cache."""
        self._lookup.clear()
        self._head.next = self._tail
        self._tail.prev = self._head

    def items(self) -> List[Tuple[K, V]]:
        """Returns list of active (key, value) pairs ordered from MRU to LRU."""
        result: List[Tuple[K, V]] = []
        curr = self._head.next

        while curr is not None and curr is not self._tail:
            next_node = curr.next
            if self._is_expired(curr):
                self._remove_node(curr)
                self._lookup.pop(curr.key, None)
            else:
                result.append((curr.key, curr.value))
            curr = next_node

        return result

    def __iter__(self) -> Iterator[K]:
        """Iterates over active keys from most recently used to least recently used."""
        for key, _ in self.items():
            yield key

