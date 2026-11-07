"""Unit tests for push-relabel maximum flow solver."""

import random
import unittest

from push_relabel import (
    FlowNetwork,
    InvalidSourceSinkError,
    NegativeCapacityError,
    NodeNotFoundError,
    PushRelabelSolver,
)


class TestPushRelabelBasics(unittest.TestCase):
    """Sanity checks for basic inputs and error handling."""

    def test_invalid_num_nodes(self) -> None:
        with self.assertRaises(ValueError):
            FlowNetwork(num_nodes=0)
        with self.assertRaises(ValueError):
            FlowNetwork(num_nodes=-3)

    def test_invalid_node_indices(self) -> None:
        net = FlowNetwork(num_nodes=3)
        with self.assertRaises(NodeNotFoundError):
            net.add_edge(source=-1, target=1, capacity=5.0)
        with self.assertRaises(NodeNotFoundError):
            net.add_edge(source=0, target=5, capacity=5.0)

    def test_negative_capacity_error(self) -> None:
        net = FlowNetwork(num_nodes=3)
        with self.assertRaises(NegativeCapacityError):
            net.add_edge(0, 1, -2.5)

    def test_identical_source_sink(self) -> None:
        net = FlowNetwork(num_nodes=3)
        with self.assertRaises(InvalidSourceSinkError):
            PushRelabelSolver(net, source=1, sink=1)

    def test_single_direct_edge(self) -> None:
        # Simple pipe: s -> t with capacity 42
        net = FlowNetwork(num_nodes=2)
        net.add_edge(0, 1, 42.0)
        solver = PushRelabelSolver(net, source=0, sink=1)
        max_flow = solver.compute_max_flow()

        self.assertAlmostEqual(max_flow, 42.0)
        self.assertTrue(solver.verify_optimality())

        cut_s, cut_t = solver.compute_min_cut()
        self.assertIn(0, cut_s)
        self.assertIn(1, cut_t)

    def test_disconnected_graph(self) -> None:
        # Source cannot reach sink
        net = FlowNetwork(num_nodes=4)
        net.add_edge(0, 1, 10.0)
        net.add_edge(2, 3, 10.0)
        solver = PushRelabelSolver(net, source=0, sink=3)
        max_flow = solver.compute_max_flow()

        self.assertAlmostEqual(max_flow, 0.0)
        self.assertTrue(solver.verify_optimality())

        cut_s, cut_t = solver.compute_min_cut()
        self.assertIn(0, cut_s)
        self.assertIn(3, cut_t)


class TestClassicFlowNetworks(unittest.TestCase):
    """Textbook benchmark topologies."""

    def test_diamond_network(self) -> None:
        # Diamond graph:
        #   0 -> 1 (cap 3), 0 -> 2 (cap 1)
        #   1 -> 3 (cap 2), 2 -> 3 (cap 4)
        # Expected max flow = 2 + 1 = 3
        net = FlowNetwork(num_nodes=4)
        net.add_edge(0, 1, 3.0)
        net.add_edge(0, 2, 1.0)
        net.add_edge(1, 3, 2.0)
        net.add_edge(2, 3, 4.0)

        solver = PushRelabelSolver(net, source=0, sink=3)
        max_flow = solver.compute_max_flow()

        self.assertAlmostEqual(max_flow, 3.0)
        self.assertTrue(solver.verify_optimality())

        # Check max-flow equals min-cut capacity
        cut_s, cut_t = solver.compute_min_cut()
        cut_capacity = sum(
            e.capacity
            for u in cut_s
            for e in net.adjacency[u]
            if e.target in cut_t and e.capacity > 0
        )
        self.assertAlmostEqual(cut_capacity, max_flow)

    def test_clrs_textbook_network(self) -> None:
        # Standard example from CLRS Chapter 26 (Flow Networks)
        # Known max flow is 23
        net = FlowNetwork(num_nodes=6)
        s, v1, v2, v3, v4, t = 0, 1, 2, 3, 4, 5
        edges = [
            (s, v1, 16.0),
            (s, v2, 13.0),
            (v1, v3, 12.0),
            (v2, v1, 4.0),
            (v2, v4, 14.0),
            (v3, v2, 9.0),
            (v3, t, 20.0),
            (v4, v3, 7.0),
            (v4, t, 4.0),
        ]
        for u, v, cap in edges:
            net.add_edge(u, v, cap)

        solver = PushRelabelSolver(net, source=s, sink=t)
        max_flow = solver.compute_max_flow()

        self.assertAlmostEqual(max_flow, 23.0)
        self.assertTrue(solver.verify_optimality())

        cut_s, cut_t = solver.compute_min_cut()
        cut_cap = sum(
            e.capacity
            for u in cut_s
            for e in net.adjacency[u]
            if e.target in cut_t and e.capacity > 0
        )
        self.assertAlmostEqual(cut_cap, 23.0)

    def test_antiparallel_edges(self) -> None:
        # Graph with antiparallel edges between nodes 1 and 2
        net = FlowNetwork(num_nodes=4)
        net.add_edge(0, 1, 10.0)
        net.add_edge(1, 2, 1.0)
        net.add_edge(2, 1, 1.0)
        net.add_edge(2, 3, 10.0)

        solver = PushRelabelSolver(net, source=0, sink=3)
        max_flow = solver.compute_max_flow()

        self.assertAlmostEqual(max_flow, 1.0)
        self.assertTrue(solver.verify_optimality())


class TestRandomizedStressAndInvariants(unittest.TestCase):
    """Stress testing on random graphs to verify flow conservation and cut duality."""

    def test_random_dag_conservation_and_cut(self) -> None:
        random.seed(42)
        n = 15
        net = FlowNetwork(num_nodes=n)
        s, t = 0, n - 1

        for u in range(n - 1):
            for v in range(u + 1, n):
                if random.random() < 0.4:
                    cap = round(random.uniform(1.0, 50.0), 2)
                    net.add_edge(u, v, cap)

        solver = PushRelabelSolver(net, source=s, sink=t)
        max_flow = solver.compute_max_flow()

        # Invariant 1: flow conservation holds everywhere
        self.assertTrue(solver.verify_optimality())

        # Invariant 2: cut capacity matches max flow
        cut_s, cut_t = solver.compute_min_cut()
        self.assertIn(s, cut_s)
        self.assertIn(t, cut_t)

        cut_capacity = sum(
            e.capacity
            for u in cut_s
            for e in net.adjacency[u]
            if e.target in cut_t and e.capacity > 0
        )
        self.assertAlmostEqual(max_flow, cut_capacity, places=5)


if __name__ == "__main__":
    unittest.main()
