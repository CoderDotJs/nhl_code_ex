"""Unit tests for LRU Cache with doubly linked list and TTL support."""

import time
import unittest

from lru_cache import LRUCache


class TestLRUCacheBasics(unittest.TestCase):
    """Sanity checks, capacity validation, and basic get/put."""

    def test_invalid_capacity(self) -> None:
        with self.assertRaises(ValueError):
            LRUCache(capacity=0)
        with self.assertRaises(ValueError):
            LRUCache(capacity=-5)

    def test_basic_put_and_get(self) -> None:
        cache: LRUCache[str, int] = LRUCache(capacity=3)
        cache.put("x", 10)
        cache.put("y", 20)

        self.assertEqual(cache.get("x"), 10)
        self.assertEqual(cache.get("y"), 20)
        self.assertIsNone(cache.get("missing"))
        self.assertEqual(cache.get("missing", default=-1), -1)
        self.assertEqual(len(cache), 2)


class TestLRUEviction(unittest.TestCase):
    """Verifies eviction order and access refresh behavior."""

    def test_evicts_least_recently_used(self) -> None:
        # Capacity 2: put A, put B, put C -> A must be evicted
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.put("C", 3)

        self.assertIsNone(cache.get("A"))
        self.assertEqual(cache.get("B"), 2)
        self.assertEqual(cache.get("C"), 3)
        self.assertEqual(len(cache), 2)

    def test_get_refreshes_mru_order(self) -> None:
        # Capacity 2: put A, put B.
        # Access A (making A MRU, B LRU).
        # Put C -> B must be evicted, A must stay!
        cache: LRUCache[str, str] = LRUCache(capacity=2)
        cache.put("A", "alpha")
        cache.put("B", "beta")

        # Refresh A
        self.assertEqual(cache.get("A"), "alpha")

        # Insert C (should evict B)
        cache.put("C", "gamma")

        self.assertEqual(cache.get("A"), "alpha")
        self.assertIsNone(cache.get("B"))
        self.assertEqual(cache.get("C"), "gamma")

    def test_update_existing_key(self) -> None:
        # Updating an existing key should change its value and refresh it
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.put("A", 100)  # Overwrite A, moving it to MRU

        self.assertEqual(len(cache), 2)

        cache.put("C", 3)  # Should evict B, not A

        self.assertEqual(cache.get("A"), 100)
        self.assertIsNone(cache.get("B"))
        self.assertEqual(cache.get("C"), 3)

    def test_capacity_one_edge_case(self) -> None:
        cache: LRUCache[int, str] = LRUCache(capacity=1)
        cache.put(1, "one")
        self.assertEqual(cache.get(1), "one")

        cache.put(2, "two")
        self.assertIsNone(cache.get(1))
        self.assertEqual(cache.get(2), "two")
        self.assertEqual(len(cache), 1)

    def test_peek_does_not_refresh_lru(self) -> None:
        # Peeking at A shouldn't refresh it; B remains MRU
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("A", 1)
        cache.put("B", 2)

        # Peek A (A is still LRU)
        val = cache.peek("A")
        self.assertEqual(val, 1)

        # Put C: A must be evicted because it was not refreshed
        cache.put("C", 3)
        self.assertIsNone(cache.get("A"))
        self.assertEqual(cache.get("B"), 2)
        self.assertEqual(cache.get("C"), 3)


class TestDeletionAndIteration(unittest.TestCase):
    """Checks explicit item deletion, clearing, and MRU-to-LRU iteration."""

    def test_delete_and_contains(self) -> None:
        cache: LRUCache[str, int] = LRUCache(capacity=3)
        cache.put("k1", 1)
        cache.put("k2", 2)

        self.assertIn("k1", cache)
        self.assertTrue(cache.delete("k1"))
        self.assertNotIn("k1", cache)
        self.assertFalse(cache.delete("k1"))  # Already deleted
        self.assertEqual(len(cache), 1)

    def test_clear(self) -> None:
        cache: LRUCache[int, int] = LRUCache(capacity=3)
        cache.put(1, 10)
        cache.put(2, 20)
        cache.clear()

        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(1))

    def test_iteration_order(self) -> None:
        cache: LRUCache[str, int] = LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Order is currently: c (MRU), b, a (LRU)
        self.assertEqual(list(cache), ["c", "b", "a"])

        # Access b
        cache.get("b")
        # Order becomes: b (MRU), c, a (LRU)
        self.assertEqual(list(cache), ["b", "c", "a"])


class TestTTLExpiration(unittest.TestCase):
    """Checks TTL (time-to-live) expiration behavior."""

    def test_item_ttl_expiration(self) -> None:
        cache: LRUCache[str, str] = LRUCache(capacity=2)
        # Put item with 0.05 second TTL
        cache.put("short_lived", "data", ttl=0.05)
        cache.put("permanent", "safe")

        self.assertEqual(cache.get("short_lived"), "data")
        self.assertIn("short_lived", cache)

        # Sleep past expiration
        time.sleep(0.06)

        self.assertIsNone(cache.get("short_lived"))
        self.assertNotIn("short_lived", cache)
        self.assertEqual(cache.get("permanent"), "safe")


if __name__ == "__main__":
    unittest.main()

