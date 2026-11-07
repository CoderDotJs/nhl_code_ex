from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

T = TypeVar("T")


class Graph(Protocol[T]):
    def neighbors(self, node: T) -> Iterable[Tuple[T, float]]:
        """Yields (neighbor_node, edge_weight) pairs for a given node."""
        ...


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[T]):
    """Container holding pathfinding outcome, total path cost, and exploration metrics."""

    path: List[T]
    cost: float
    nodes_explored: int


# --- Standard Heuristic Functions for 2D Grids ---


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """L1 distance heuristic (admissible for 4-directional grid movement)."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """L2 distance heuristic (straight-line distance, always admissible in Euclidean space)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def octile_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Admissible heuristic for 8-directional movement with diagonal cost sqrt(2)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)



def astar_search(
    graph: Graph[T],
    start: T,
    goal: T,
    heuristic: Optional[Callable[[T, T], float]] = None,
) -> Optional[SearchResult[T]]:
    if start == goal:
        return SearchResult(path=[start], cost=0.0, nodes_explored=1)

    h_fn = heuristic if heuristic is not None else (lambda u, v: 0.0)

    # Priority queue stores tuples of: (f_score, tiebreaker_counter, node)
    # tiebreaker avoids comparing node objects directly when f_scores match.
    open_set: List[Tuple[float, int, T]] = []
    tiebreaker = 0

    heapq.heappush(open_set, (h_fn(start, goal), tiebreaker, start))

    came_from: Dict[T, T] = {}
    g_score: Dict[T, float] = {start: 0.0}
    closed_set: Set[T] = set()

    nodes_explored = 0

    while open_set:
        current_f, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_explored += 1

        # Goal reached: reconstruct shortest path
        if current == goal:
            path = _reconstruct_path(came_from, current)
            return SearchResult(
                path=path,
                cost=g_score[current],
                nodes_explored=nodes_explored,
            )

        for neighbor, edge_cost in graph.neighbors(current):
            if neighbor in closed_set:
                continue

            if edge_cost < 0:
                raise ValueError(
                    f"A* requires non-negative edge costs, received {edge_cost}"
                )

            tentative_g = g_score[current] + edge_cost

            # If this is the best path to neighbor found so far
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h_fn(neighbor, goal)

                tiebreaker += 1
                heapq.heappush(open_set, (f_score, tiebreaker, neighbor))

    return None


def _reconstruct_path(came_from: Dict[T, T], current: T) -> List[T]:
    """Reconstructs the path list [start, ..., current] from parent pointers."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


class GridGraph:
    """A 2D obstacle grid graph supporting 4-way or 8-way movement."""

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        allow_diagonal: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles is not None else set()
        self.allow_diagonal = allow_diagonal

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, pos: Tuple[int, int]) -> bool:
        return pos not in self.obstacles

    def neighbors(self, node: Tuple[int, int]) -> Iterable[Tuple[Tuple[int, int], float]]:
        x, y = node

        # 4 cardinal directions (cost 1.0)
        cardinals = [
            ((x + 1, y), 1.0),
            ((x - 1, y), 1.0),
            ((x, y + 1), 1.0),
            ((x, y - 1), 1.0),
        ]

        for next_pos, cost in cardinals:
            if self.in_bounds(next_pos) and self.is_passable(next_pos):
                yield (next_pos, cost)

        if self.allow_diagonal:
            # 4 diagonal directions (cost sqrt(2) ~ 1.414)
            diagonals = [
                ((x + 1, y + 1), math.sqrt(2)),
                ((x - 1, y + 1), math.sqrt(2)),
                ((x + 1, y - 1), math.sqrt(2)),
                ((x - 1, y - 1), math.sqrt(2)),
            ]
            for next_pos, cost in diagonals:
                if self.in_bounds(next_pos) and self.is_passable(next_pos):
                    # Prevent corner cutting through adjacent blocked diagonal walls
                    nx, ny = next_pos
                    if self.is_passable((x, ny)) and self.is_passable((nx, y)):
                        yield (next_pos, cost)

