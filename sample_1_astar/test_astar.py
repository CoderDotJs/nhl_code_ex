"""Unit tests for A* pathfinding algorithm."""

import math
import unittest
from typing import Dict, Iterable, List, Tuple

from astar import (
    GridGraph,
    astar_search,
    euclidean_distance,
    manhattan_distance,
    octile_distance,
)


class SimpleWeightedGraph:
    """Mock general graph for testing non-grid topologies."""

    def __init__(self) -> None:
        self.adj: Dict[str, List[Tuple[str, float]]] = {}

    def add_edge(self, u: str, v: str, weight: float) -> None:
        self.adj.setdefault(u, []).append((v, weight))

    def neighbors(self, node: str) -> Iterable[Tuple[str, float]]:
        return self.adj.get(node, [])


class TestAStarBasics(unittest.TestCase):
    """Basic pathfinding sanity checks on 2D grids."""

    def test_start_equals_goal(self) -> None:
        grid = GridGraph(width=5, height=5)
        res = astar_search(grid, start=(2, 2), goal=(2, 2))
        self.assertIsNotNone(res)
        self.assertEqual(res.path, [(2, 2)])
        self.assertEqual(res.cost, 0.0)
        self.assertEqual(res.nodes_explored, 1)

    def test_straight_line_path(self) -> None:
        grid = GridGraph(width=10, height=10)
        res = astar_search(
            grid,
            start=(0, 0),
            goal=(0, 4),
            heuristic=manhattan_distance,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.cost, 4.0)
        self.assertEqual(len(res.path), 5)
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (0, 4))

    def test_obstacle_avoidance(self) -> None:
        # 5x5 grid with vertical wall at x=2, y in [0, 1, 2]
        # Start at (0, 1), Goal at (4, 1)
        obstacles = {(2, 0), (2, 1), (2, 2)}
        grid = GridGraph(width=5, height=5, obstacles=obstacles)

        res = astar_search(
            grid,
            start=(0, 1),
            goal=(4, 1),
            heuristic=manhattan_distance,
        )
        self.assertIsNotNone(res)
        # Must walk around the wall (down through y=3 or above)
        for obstacle in obstacles:
            self.assertNotIn(obstacle, res.path)

        self.assertEqual(res.path[0], (0, 1))
        self.assertEqual(res.path[-1], (4, 1))

    def test_unreachable_goal(self) -> None:
        # Goal completely surrounded by obstacles
        obstacles = {(4, 3), (3, 4)}
        grid = GridGraph(width=5, height=5, obstacles=obstacles)
        res = astar_search(
            grid,
            start=(0, 0),
            goal=(4, 4),
            heuristic=manhattan_distance,
        )
        self.assertIsNone(res)


class TestAStarDiagonalsAndCornerCutting(unittest.TestCase):
    """8-directional movement and diagonal corner-cutting prevention."""

    def test_diagonal_movement(self) -> None:
        grid = GridGraph(width=4, height=4, allow_diagonal=True)
        res = astar_search(
            grid,
            start=(0, 0),
            goal=(3, 3),
            heuristic=octile_distance,
        )
        self.assertIsNotNone(res)
        # Straight diagonal of 3 steps = 3 * sqrt(2)
        expected_cost = 3.0 * math.sqrt(2)
        self.assertAlmostEqual(res.cost, expected_cost)
        self.assertEqual(len(res.path), 4)

    def test_prevent_corner_cutting(self) -> None:
        # Corner pinch:
        # (0, 1) is blocked and (1, 0) is blocked.
        # Diagonal move from (0, 0) to (1, 1) must be forbidden!
        obstacles = {(0, 1), (1, 0)}
        grid = GridGraph(width=3, height=3, obstacles=obstacles, allow_diagonal=True)
        res = astar_search(
            grid,
            start=(0, 0),
            goal=(1, 1),
            heuristic=octile_distance,
        )
        # Should be None because (0, 0) cannot escape
        self.assertIsNone(res)


class TestGeneralWeightedGraph(unittest.TestCase):
    """Testing A* over general non-grid graphs."""

    def test_romania_style_graph(self) -> None:
        # Graph with alternate paths:
        # A -> B (cost 4), B -> D (cost 5) = total 9
        # A -> C (cost 2), C -> D (cost 10) = total 12
        # Shortest is A -> B -> D with cost 9
        graph = SimpleWeightedGraph()
        graph.add_edge("A", "B", 4.0)
        graph.add_edge("B", "D", 5.0)
        graph.add_edge("A", "C", 2.0)
        graph.add_edge("C", "D", 10.0)

        # Admissible heuristic (underestimates)
        heuristic_map = {"A": 7.0, "B": 4.0, "C": 8.0, "D": 0.0}

        res = astar_search(
            graph,
            start="A",
            goal="D",
            heuristic=lambda u, v: heuristic_map.get(u, 0.0),
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.path, ["A", "B", "D"])
        self.assertEqual(res.cost, 9.0)

    def test_negative_weight_rejection(self) -> None:
        graph = SimpleWeightedGraph()
        graph.add_edge("A", "B", -1.0)
        with self.assertRaises(ValueError):
            astar_search(graph, start="A", goal="B")

    def test_heuristic_pruning_efficiency(self) -> None:
        # Compare explored nodes between informed A* and uninformed Dijkstra (h=0)
        grid = GridGraph(width=20, height=20)
        start, goal = (0, 0), (18, 18)

        dijkstra_res = astar_search(grid, start, goal, heuristic=None)
        astar_res = astar_search(grid, start, goal, heuristic=manhattan_distance)

        self.assertIsNotNone(dijkstra_res)
        self.assertIsNotNone(astar_res)
        self.assertEqual(dijkstra_res.cost, astar_res.cost)
        # A* with Manhattan heuristic must explore strictly fewer nodes on an open grid
        self.assertLess(astar_res.nodes_explored, dijkstra_res.nodes_explored)


if __name__ == "__main__":
    unittest.main()

