"""Unit tests for k-d tree spatial index."""

import math
import random
import unittest
from typing import List

from kd_tree import (
    DimensionalityError,
    EmptyTreeError,
    HyperRectangle,
    InvalidMetricError,
    KDPoint,
    KDTree,
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
)


class TestKDTreeBasics(unittest.TestCase):
    """Sanity checks on inputs, exceptions, and distance metrics."""

    def test_invalid_point_initialization(self) -> None:
        with self.assertRaises(DimensionalityError):
            KDPoint(())

        with self.assertRaises(ValueError):
            KDPoint((1.0, float("nan")))

        with self.assertRaises(ValueError):
            KDPoint((float("inf"), 2.0))

    def test_metric_calculations(self) -> None:
        p1 = (1.0, 2.0, 3.0)
        p2 = (4.0, 6.0, 3.0)

        # Euclidean: sqrt(3^2 + 4^2 + 0^2) = 5.0
        self.assertAlmostEqual(euclidean_distance(p1, p2), 5.0)

        # Manhattan: 3 + 4 + 0 = 7.0
        self.assertAlmostEqual(manhattan_distance(p1, p2), 7.0)

        # Chebyshev: max(3, 4, 0) = 4.0
        self.assertAlmostEqual(chebyshev_distance(p1, p2), 4.0)

        # Minkowski p=3: (3^3 + 4^3)^(1/3) = 91^(1/3)
        m3 = minkowski_distance(3.0)
        self.assertAlmostEqual(m3(p1, p2), 91.0 ** (1.0 / 3.0))

    def test_invalid_minkowski_order(self) -> None:
        with self.assertRaises(InvalidMetricError):
            minkowski_distance(0.5)

    def test_empty_tree_operations(self) -> None:
        with self.assertRaises(EmptyTreeError):
            KDTree.build([])

        tree: KDTree[str] = KDTree(dim=2)
        with self.assertRaises(EmptyTreeError):
            tree.nearest_neighbor((1.0, 2.0))


class TestNearestNeighbors(unittest.TestCase):
    """Nearest neighbor search on a small 2D point cloud."""

    def setUp(self) -> None:
        self.points = [
            KDPoint((2.0, 3.0), payload="A"),
            KDPoint((5.0, 4.0), payload="B"),
            KDPoint((9.0, 6.0), payload="C"),
            KDPoint((4.0, 7.0), payload="D"),
            KDPoint((8.0, 1.0), payload="E"),
            KDPoint((7.0, 2.0), payload="F"),
        ]
        self.tree = KDTree.build(self.points)

    def test_single_nearest_neighbor(self) -> None:
        # Closest to (2.1, 3.1) should be A at (2.0, 3.0)
        res = self.tree.nearest_neighbor((2.1, 3.1))
        self.assertEqual(res.point.payload, "A")
        expected_dist = math.sqrt((2.1 - 2.0) ** 2 + (3.1 - 3.0) ** 2)
        self.assertAlmostEqual(res.distance, expected_dist)

    def test_exact_point_query(self) -> None:
        # Distance to an identical point in the tree should be 0.0
        res = self.tree.nearest_neighbor((9.0, 6.0))
        self.assertEqual(res.point.payload, "C")
        self.assertAlmostEqual(res.distance, 0.0)

    def test_k_nearest_neighbors(self) -> None:
        query = (5.1, 4.1)
        results = self.tree.k_nearest_neighbors(query, k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].point.payload, "B")

        # Must be sorted ascending
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i].distance, results[i + 1].distance)

    def test_knn_with_manhattan_metric(self) -> None:
        query = (6.0, 3.0)
        results = self.tree.k_nearest_neighbors(query, k=2, metric=manhattan_distance)

        # Both (5.0, 4.0) and (7.0, 2.0) are tied at Manhattan distance 2.0
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].distance, 2.0)
        self.assertAlmostEqual(results[1].distance, 2.0)
        result_coords = {r.point.coords for r in results}
        self.assertEqual(result_coords, {(5.0, 4.0), (7.0, 2.0)})


class TestRangeAndRadiusSearch(unittest.TestCase):
    """Range boxes and radius queries."""

    def setUp(self) -> None:
        self.points = [
            KDPoint((1.0, 2.0), "P1"),
            KDPoint((3.0, 6.0), "P2"),
            KDPoint((8.0, 5.0), "P3"),
            KDPoint((4.0, 3.0), "P4"),
            KDPoint((6.0, 8.0), "P5"),
        ]
        self.tree = KDTree.build(self.points)

    def test_range_search(self) -> None:
        # Box [2.0, 7.0] x [2.0, 7.0] contains P2(3, 6) and P4(4, 3)
        box = HyperRectangle(min_bounds=(2.0, 2.0), max_bounds=(7.0, 7.0))
        results = self.tree.range_search(box)
        payloads = {pt.payload for pt in results}
        self.assertEqual(payloads, {"P2", "P4"})

    def test_radius_search(self) -> None:
        # Radius 2.0 around (3.0, 3.0) should only match P4(4, 3) at dist 1.0
        results = self.tree.radius_search(center=(3.0, 3.0), radius=2.0)
        coords = [r.point.coords for r in results]
        self.assertEqual(coords, [(4.0, 3.0)])


class TestDynamicInsertionAndRebalancing(unittest.TestCase):
    """Dynamic additions and balance maintenance."""

    def test_incremental_insertion(self) -> None:
        tree: KDTree[int] = KDTree(dim=3)
        coords_list = [(float(i), float(i * 2), float(i * 3)) for i in range(1, 25)]

        for idx, coords in enumerate(coords_list):
            tree.insert(KDPoint(coords, payload=idx))

        self.assertEqual(tree.size, 24)
        self.assertTrue(tree.verify_invariants())

        res = tree.nearest_neighbor((1.1, 2.1, 3.1))
        self.assertEqual(res.point.coords, (1.0, 2.0, 3.0))


class TestRandomizedStressAndFuzzing(unittest.TestCase):
    """Fuzzing in 5D against brute-force linear search."""

    def test_5d_randomized_knn_and_range(self) -> None:
        random.seed(1337)
        dim = 5
        num_points = 100

        points: List[KDPoint[int]] = []
        for i in range(num_points):
            coords = tuple(round(random.uniform(-100.0, 100.0), 3) for _ in range(dim))
            points.append(KDPoint(coords, payload=i))

        tree = KDTree.build(points)
        self.assertTrue(tree.verify_invariants())

        # Test 20 random queries against linear brute-force scan
        for _ in range(20):
            query = tuple(round(random.uniform(-100.0, 100.0), 3) for _ in range(dim))
            k = random.randint(1, 10)

            kd_res = tree.k_nearest_neighbors(query, k=k)
            brute_force = sorted(
                points, key=lambda pt: euclidean_distance(query, pt.coords)
            )[:k]

            for kd_item, bf_item in zip(kd_res, brute_force):
                expected = euclidean_distance(query, bf_item.coords)
                self.assertAlmostEqual(kd_item.distance, expected, places=5)


if __name__ == "__main__":
    unittest.main()
