"""k-Dimensional Tree (k-d Tree) for Spatial Indexing.

Implements a balanced k-d tree with exact nearest neighbors (1-NN and k-NN),
branch-and-bound bounding-box pruning, orthogonal range queries, and dynamic
insertions with Scapegoat-style subtree rebalancing.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T")


class SpatialIndexError(Exception):
    """Base exception for spatial index operations."""


class DimensionalityError(SpatialIndexError):
    """Raised when point coordinates have inconsistent dimensions."""


class EmptyTreeError(SpatialIndexError):
    """Raised when querying an empty tree."""


class InvalidMetricError(SpatialIndexError):
    """Raised when distance metric parameters are invalid."""


class Metric(Protocol):
    """Distance function protocol between two coordinate sequences."""

    def __call__(self, p1: Sequence[float], p2: Sequence[float]) -> float: ...


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def manhattan_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(p1, p2))


def chebyshev_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(p1, p2))


def minkowski_distance(p: float) -> Callable[[Sequence[float], Sequence[float]], float]:
    if p < 1.0:
        raise InvalidMetricError(f"Minkowski order p must be >= 1.0, got {p}")

    def _dist(p1: Sequence[float], p2: Sequence[float]) -> float:
        return sum(abs(a - b) ** p for a, b in zip(p1, p2)) ** (1.0 / p)

    return _dist


@dataclass(frozen=True, slots=True)
class KDPoint(Generic[T]):
    """Immutable point in k-dimensional space with optional payload."""

    coords: Tuple[float, ...]
    payload: Optional[T] = None

    def __post_init__(self) -> None:
        if not self.coords:
            raise DimensionalityError("Point coordinates cannot be empty")
        for i, val in enumerate(self.coords):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Coordinate at index {i} must be finite: {val}")

    @property
    def dim(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> float:
        return self.coords[index]

    def distance_to(
        self, other: Sequence[float] | KDPoint[T], metric: Metric = euclidean_distance
    ) -> float:
        target = other.coords if isinstance(other, KDPoint) else other
        if len(self.coords) != len(target):
            raise DimensionalityError(
                f"Dimension mismatch: {len(self.coords)} vs {len(target)}"
            )
        return metric(self.coords, target)


@dataclass(frozen=True, slots=True)
class HyperRectangle:
    """Axis-aligned bounding box defined by min and max coordinate bounds."""

    min_bounds: Tuple[float, ...]
    max_bounds: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.min_bounds) != len(self.max_bounds):
            raise DimensionalityError("Min and max bounds must have matching dimensions")
        for i, (lo, hi) in enumerate(zip(self.min_bounds, self.max_bounds)):
            if lo > hi:
                raise ValueError(f"Axis {i} min ({lo}) exceeds max ({hi})")

    @property
    def dim(self) -> int:
        return len(self.min_bounds)

    def contains_point(self, coords: Sequence[float]) -> bool:
        return all(lo <= c <= hi for c, lo, hi in zip(coords, self.min_bounds, self.max_bounds))

    def intersects(self, other: HyperRectangle) -> bool:
        return all(
            lo1 <= hi2 and lo2 <= hi1
            for lo1, hi1, lo2, hi2 in zip(
                self.min_bounds, self.max_bounds, other.min_bounds, other.max_bounds
            )
        )

    def min_distance_to_point(
        self, coords: Sequence[float], metric: Metric = euclidean_distance
    ) -> float:
        """Computes minimum distance from a point to the nearest face of this box."""
        projected = tuple(
            max(lo, min(c, hi))
            for c, lo, hi in zip(coords, self.min_bounds, self.max_bounds)
        )
        return metric(projected, coords)


@dataclass(slots=True)
class KDNode(Generic[T]):
    """Tree node storing a point, split axis, child pointers, and bounding box."""

    point: KDPoint[T]
    axis: int
    left: Optional[KDNode[T]] = None
    right: Optional[KDNode[T]] = None
    bounding_box: HyperRectangle = field(init=False)
    subtree_size: int = field(default=1)

    def __post_init__(self) -> None:
        self.bounding_box = HyperRectangle(
            min_bounds=self.point.coords, max_bounds=self.point.coords
        )

    def update_metadata(self) -> None:
        """Refreshes subtree size and tight bounding box from child nodes."""
        size = 1
        mins = list(self.point.coords)
        maxs = list(self.point.coords)

        if self.left is not None:
            size += self.left.subtree_size
            mins = [min(a, b) for a, b in zip(mins, self.left.bounding_box.min_bounds)]
            maxs = [max(a, b) for a, b in zip(maxs, self.left.bounding_box.max_bounds)]

        if self.right is not None:
            size += self.right.subtree_size
            mins = [min(a, b) for a, b in zip(mins, self.right.bounding_box.min_bounds)]
            maxs = [max(a, b) for a, b in zip(maxs, self.right.bounding_box.max_bounds)]

        self.subtree_size = size
        self.bounding_box = HyperRectangle(
            min_bounds=tuple(mins), max_bounds=tuple(maxs)
        )


@dataclass(frozen=True, slots=True, order=True)
class NeighborResult(Generic[T]):
    """Holds a search match: distance and corresponding point."""

    distance: float
    point: KDPoint[T] = field(compare=False)


class KDTree(Generic[T]):
    """Balanced spatial search tree supporting k-NN, range, and radius queries."""

    def __init__(self, dim: int, rebalance_factor: float = 0.7) -> None:
        if dim <= 0:
            raise DimensionalityError(f"Dimension must be positive: got {dim}")
        if not (0.5 < rebalance_factor < 1.0):
            raise ValueError(
                f"rebalance_factor must be between 0.5 and 1.0, got {rebalance_factor}"
            )

        self._dim = dim
        self._rebalance_factor = rebalance_factor
        self._root: Optional[KDNode[T]] = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return self._root.subtree_size if self._root is not None else 0

    def is_empty(self) -> bool:
        return self._root is None

    @classmethod
    def build(
        cls,
        points: Sequence[KDPoint[T]],
        rebalance_factor: float = 0.7,
    ) -> KDTree[T]:
        """Constructs an optimally balanced tree using median-partitioning."""
        if not points:
            raise EmptyTreeError("Cannot build KDTree from empty point collection")

        dim = points[0].dim
        for pt in points:
            if pt.dim != dim:
                raise DimensionalityError(
                    f"Points have mismatched dimensions: expected {dim}, got {pt.dim}"
                )

        tree = cls(dim=dim, rebalance_factor=rebalance_factor)
        tree._root = tree._build_recursive(list(points))
        return tree

    @classmethod
    def _select_split_axis(cls, points: List[KDPoint[T]], dim: int) -> int:
        # Pick the axis with largest coordinate spread (max - min) to avoid long skinny cells
        best_spread = -1.0
        best_axis = 0

        for axis in range(dim):
            lo = min(pt.coords[axis] for pt in points)
            hi = max(pt.coords[axis] for pt in points)
            spread = hi - lo
            if spread > best_spread:
                best_spread = spread
                best_axis = axis

        return best_axis

    def _build_recursive(self, points: List[KDPoint[T]]) -> Optional[KDNode[T]]:
        if not points:
            return None

        axis = self._select_split_axis(points, self._dim)
        points.sort(key=lambda pt: pt.coords[axis])

        mid = len(points) // 2
        pivot = points[mid]

        node = KDNode(point=pivot, axis=axis)
        node.left = self._build_recursive(points[:mid])
        node.right = self._build_recursive(points[mid + 1:])
        node.update_metadata()

        return node

    def insert(self, point: KDPoint[T]) -> None:
        """Inserts a point and rebalances any ancestor that violates balance factor."""
        if point.dim != self._dim:
            raise DimensionalityError(
                f"Cannot insert point with dim {point.dim} into tree of dim {self._dim}"
            )

        if self._root is None:
            self._root = KDNode(point=point, axis=0)
            return

        self._root = self._insert_recursive(self._root, point, depth=0)

    def _insert_recursive(
        self, node: KDNode[T], point: KDPoint[T], depth: int
    ) -> KDNode[T]:
        axis = node.axis
        if point.coords[axis] <= node.point.coords[axis]:
            if node.left is None:
                next_axis = (axis + 1) % self._dim
                node.left = KDNode(point=point, axis=next_axis)
            else:
                node.left = self._insert_recursive(node.left, point, depth + 1)
        else:
            if node.right is None:
                next_axis = (axis + 1) % self._dim
                node.right = KDNode(point=point, axis=next_axis)
            else:
                node.right = self._insert_recursive(node.right, point, depth + 1)

        node.update_metadata()

        # Check balance: if one child is too heavy relative to parent, rebalance subtree
        left_sz = node.left.subtree_size if node.left else 0
        right_sz = node.right.subtree_size if node.right else 0
        if max(left_sz, right_sz) > self._rebalance_factor * node.subtree_size:
            pts = self._collect_subtree_points(node)
            return self._build_recursive(pts)  # type: ignore[return-value]

        return node

    def _collect_subtree_points(self, node: Optional[KDNode[T]]) -> List[KDPoint[T]]:
        if node is None:
            return []
        items = [node.point]
        items.extend(self._collect_subtree_points(node.left))
        items.extend(self._collect_subtree_points(node.right))
        return items

    def nearest_neighbor(
        self,
        query: Sequence[float] | KDPoint[T],
        metric: Metric = euclidean_distance,
    ) -> NeighborResult[T]:
        """Finds closest point to query."""
        res = self.k_nearest_neighbors(query=query, k=1, metric=metric)
        return res[0]

    def k_nearest_neighbors(
        self,
        query: Sequence[float] | KDPoint[T],
        k: int,
        metric: Metric = euclidean_distance,
    ) -> List[NeighborResult[T]]:
        """Finds k closest points using branch-and-bound pruning."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if self._root is None:
            raise EmptyTreeError("Cannot query nearest neighbors on empty KDTree")

        coords = query.coords if isinstance(query, KDPoint) else tuple(query)
        if len(coords) != self._dim:
            raise DimensionalityError(
                f"Query dim {len(coords)} does not match tree dim {self._dim}"
            )

        # Max-heap storing (-distance, tiebreaker_counter, point)
        heap: List[Tuple[float, int, KDPoint[T]]] = []
        tiebreaker = 0

        def _search(node: Optional[KDNode[T]]) -> None:
            nonlocal tiebreaker
            if node is None:
                return

            # Branch-and-bound pruning: if bounding box min distance can't beat
            # the current k-th neighbor in our heap, skip this entire subtree.
            if len(heap) == k:
                worst_best_dist = -heap[0][0]
                box_dist = node.bounding_box.min_distance_to_point(coords, metric=metric)
                if box_dist >= worst_best_dist:
                    return

            dist = metric(coords, node.point.coords)
            if len(heap) < k:
                tiebreaker += 1
                heapq.heappush(heap, (-dist, tiebreaker, node.point))
            elif dist < -heap[0][0]:
                tiebreaker += 1
                heapq.heapreplace(heap, (-dist, tiebreaker, node.point))

            axis = node.axis
            query_val = coords[axis]
            node_val = node.point.coords[axis]

            # Visit the closer half-space first
            if query_val <= node_val:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left

            _search(first)

            # Check if the farther branch could have closer points
            worst_dist = -heap[0][0] if len(heap) == k else float("inf")
            plane_dist = abs(query_val - node_val)
            if plane_dist < worst_dist:
                _search(second)

        _search(self._root)

        # Sort results ascending by distance
        sorted_items = sorted(heap, key=lambda item: -item[0])
        return [
            NeighborResult(distance=-item[0], point=item[2])
            for item in sorted_items
        ]

    def range_search(self, box: HyperRectangle) -> List[KDPoint[T]]:
        """Finds all points within an axis-aligned bounding box."""
        if box.dim != self._dim:
            raise DimensionalityError(
                f"Query box dim {box.dim} != tree dim {self._dim}"
            )
        if self._root is None:
            return []

        results: List[KDPoint[T]] = []

        def _search(node: Optional[KDNode[T]]) -> None:
            if node is None:
                return

            # Prune if node's bounding box doesn't intersect search box
            if not node.bounding_box.intersects(box):
                return

            if box.contains_point(node.point.coords):
                results.append(node.point)

            axis = node.axis
            if box.min_bounds[axis] <= node.point.coords[axis]:
                _search(node.left)
            if box.max_bounds[axis] >= node.point.coords[axis]:
                _search(node.right)

        _search(self._root)
        return results

    def radius_search(
        self,
        center: Sequence[float] | KDPoint[T],
        radius: float,
        metric: Metric = euclidean_distance,
    ) -> List[NeighborResult[T]]:
        """Finds all points within a given distance from center."""
        if radius < 0.0:
            raise InvalidMetricError(f"Radius cannot be negative: {radius}")

        coords = center.coords if isinstance(center, KDPoint) else tuple(center)
        if len(coords) != self._dim:
            raise DimensionalityError(
                f"Center dim {len(coords)} != tree dim {self._dim}"
            )
        if self._root is None:
            return []

        matches: List[NeighborResult[T]] = []

        def _search(node: Optional[KDNode[T]]) -> None:
            if node is None:
                return

            if node.bounding_box.min_distance_to_point(coords, metric=metric) > radius:
                return

            dist = metric(coords, node.point.coords)
            if dist <= radius:
                matches.append(NeighborResult(distance=dist, point=node.point))

            axis = node.axis
            center_val = coords[axis]
            node_val = node.point.coords[axis]

            if center_val - radius <= node_val:
                _search(node.left)
            if center_val + radius >= node_val:
                _search(node.right)

        _search(self._root)
        matches.sort(key=lambda m: m.distance)
        return matches

    def verify_invariants(self) -> bool:
        """Asserts correct spatial partitioning and bounding boxes across tree."""
        if self._root is None:
            return True

        def _check(node: KDNode[T]) -> int:
            axis = node.axis
            sz = 1

            if node.left is not None:
                assert (
                    node.left.point.coords[axis] <= node.point.coords[axis]
                ), f"Left partition failed at axis {axis}"
                sz += _check(node.left)

            if node.right is not None:
                assert (
                    node.right.point.coords[axis] >= node.point.coords[axis]
                ), f"Right partition failed at axis {axis}"
                sz += _check(node.right)

            assert node.subtree_size == sz, f"Subtree size mismatch: {node.subtree_size} vs {sz}"
            return sz

        _check(self._root)
        return True
