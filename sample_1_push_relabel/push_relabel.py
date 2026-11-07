"""Goldberg-Tarjan Push-Relabel Maximum Flow Algorithm.

A pure Python implementation of the Goldberg-Tarjan push-relabel algorithm
for maximum s-t flow and minimum s-t cut, optimized with highest-label
selection, gap relabeling, and periodic global relabeling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


class NetworkFlowError(Exception):
    """Base exception for network flow errors."""


class NodeNotFoundError(NetworkFlowError):
    """Raised when an operation references an out-of-bounds node."""


class NegativeCapacityError(NetworkFlowError):
    """Raised when an edge capacity is negative."""


class InvalidSourceSinkError(NetworkFlowError):
    """Raised when source or sink configurations are invalid."""


class FlowConservationError(NetworkFlowError):
    """Raised when flow conservation or capacity constraints are violated."""


@dataclass(slots=True)
class FlowEdge:
    """Directed edge in the flow network paired with its residual back-edge."""

    source: int
    target: int
    capacity: float
    flow: float
    residual_index: int

    @property
    def residual_capacity(self) -> float:
        return self.capacity - self.flow


class FlowNetwork:
    """Adjacency-list based flow network supporting real-valued capacities."""

    def __init__(self, num_nodes: int) -> None:
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")

        self._num_nodes = num_nodes
        self._adjacency: List[List[FlowEdge]] = [[] for _ in range(num_nodes)]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def adjacency(self) -> List[List[FlowEdge]]:
        return self._adjacency

    def add_edge(self, source: int, target: int, capacity: float) -> None:
        """Adds a directed edge and its paired residual reverse edge."""
        if not (0 <= source < self._num_nodes):
            raise NodeNotFoundError(f"Source node {source} is out of bounds")
        if not (0 <= target < self._num_nodes):
            raise NodeNotFoundError(f"Target node {target} is out of bounds")
        if capacity < 0.0:
            raise NegativeCapacityError(f"Edge capacity cannot be negative: {capacity}")

        fwd_idx = len(self._adjacency[source])
        rev_idx = len(self._adjacency[target])

        forward_edge = FlowEdge(
            source=source,
            target=target,
            capacity=float(capacity),
            flow=0.0,
            residual_index=rev_idx,
        )
        backward_edge = FlowEdge(
            source=target,
            target=source,
            capacity=0.0,
            flow=0.0,
            residual_index=fwd_idx,
        )

        self._adjacency[source].append(forward_edge)
        self._adjacency[target].append(backward_edge)

    def get_edge(self, source: int, edge_index: int) -> FlowEdge:
        return self._adjacency[source][edge_index]

    def get_residual_edge(self, edge: FlowEdge) -> FlowEdge:
        return self._adjacency[edge.target][edge.residual_index]


class PushRelabelSolver:
    """Highest-label push-relabel solver with gap and global relabeling heuristics.

    Complexity:
      - Time: O(V^2 * sqrt(E)) worst-case, typically near O(E^1.2) in practice.
      - Space: O(V + E).
    """

    def __init__(self, network: FlowNetwork, source: int, sink: int) -> None:
        if not (0 <= source < network.num_nodes):
            raise InvalidSourceSinkError(f"Source node {source} out of bounds")
        if not (0 <= sink < network.num_nodes):
            raise InvalidSourceSinkError(f"Sink node {sink} out of bounds")
        if source == sink:
            raise InvalidSourceSinkError("Source and sink nodes must be distinct")

        self.network = network
        self.source = source
        self.sink = sink
        self.n = network.num_nodes

        # Core state
        self.excess: List[float] = [0.0] * self.n
        self.height: List[int] = [0] * self.n
        self.height_count: List[int] = [0] * (2 * self.n + 1)
        self.current_edge_index: List[int] = [0] * self.n

        # Buckets for active nodes: active_buckets[h] holds nodes with excess > 0 at height h
        self.active_buckets: List[List[int]] = [[] for _ in range(2 * self.n + 1)]
        self.max_active_height: int = 0

        # Frequency trigger for periodic global relabeling
        self._work_counter = 0
        self._global_relabel_frequency = 4 * self.n

    def _mark_active(self, u: int) -> None:
        """Pushes node u into active bucket if it has excess and isn't source or sink."""
        if u != self.source and u != self.sink and self.excess[u] > 1e-9:
            h = self.height[u]
            self.active_buckets[h].append(u)
            if h > self.max_active_height:
                self.max_active_height = h

    def _push(self, edge: FlowEdge) -> float:
        """Pushes excess flow through residual edge (u, v)."""
        u, v = edge.source, edge.target
        send = min(self.excess[u], edge.residual_capacity)
        if send <= 1e-12:
            return 0.0

        edge.flow += send
        rev = self.network.get_residual_edge(edge)
        rev.flow -= send

        self.excess[u] -= send
        self.excess[v] += send

        # If v was inactive and now has excess, add to bucket
        if v != self.source and v != self.sink and self.excess[v] - send <= 1e-9:
            self._mark_active(v)

        return send

    def _relabel(self, u: int) -> None:
        """Lifts height of u to 1 + min{height[v]} over available residual edges."""
        self._work_counter += 1
        old_height = self.height[u]
        min_h = 2 * self.n

        for edge in self.network.adjacency[u]:
            if edge.residual_capacity > 1e-9:
                min_h = min(min_h, self.height[edge.target])

        new_height = min_h + 1
        self.height_count[old_height] -= 1

        # Gap heuristic: if old_height is now completely empty, all nodes higher
        # up (below n) are stranded from t in the residual graph. Lift them to n+1.
        if 0 < old_height < self.n and self.height_count[old_height] == 0:
            self._apply_gap_heuristic(old_height)
            return

        self.height[u] = new_height
        self.height_count[new_height] += 1
        self.current_edge_index[u] = 0

        if new_height < self.n:
            self._mark_active(u)

    def _apply_gap_heuristic(self, gap_height: int) -> None:
        for v in range(self.n):
            if v == self.source or v == self.sink:
                continue
            h = self.height[v]
            if gap_height < h < self.n:
                self.height_count[h] -= 1
                new_h = max(h, self.n + 1)
                self.height[v] = new_h
                self.height_count[new_h] += 1

    def _discharge(self, u: int) -> None:
        """Repeatedly pushes from u until its excess is drained or it gets relabeled."""
        adj = self.network.adjacency[u]
        deg = len(adj)

        while self.excess[u] > 1e-9:
            if self.current_edge_index[u] < deg:
                edge = adj[self.current_edge_index[u]]
                if (
                    edge.residual_capacity > 1e-9
                    and self.height[u] == self.height[edge.target] + 1
                ):
                    self._push(edge)
                else:
                    self.current_edge_index[u] += 1
            else:
                self._relabel(u)
                if self.height[u] >= self.n:
                    break

    def _global_relabel(self) -> None:
        """Backward BFS from sink in residual graph to recalculate exact distances."""
        for i in range(len(self.height)):
            self.height[i] = self.n
        for i in range(len(self.height_count)):
            self.height_count[i] = 0
        for bucket in self.active_buckets:
            bucket.clear()

        queue: deque[int] = deque([self.sink])
        self.height[self.sink] = 0
        self.height_count[0] = 1

        while queue:
            curr = queue.popleft()
            h_next = self.height[curr] + 1
            for edge in self.network.adjacency[curr]:
                rev = self.network.get_residual_edge(edge)
                if rev.residual_capacity > 1e-9 and self.height[rev.source] == self.n:
                    self.height[rev.source] = h_next
                    self.height_count[h_next] += 1
                    queue.append(rev.source)

        self.height[self.source] = self.n
        self.height_count[self.n] += 1

        # Re-populate active buckets
        self.max_active_height = 0
        for v in range(self.n):
            if v != self.source and v != self.sink and self.excess[v] > 1e-9:
                if self.height[v] < self.n:
                    self._mark_active(v)

        self._work_counter = 0

    def _convert_preflow_to_flow(self) -> None:
        """Returns leftover excess preflow at internal nodes back to the source.

        Cancels flow along backwards paths to source and breaks any flow circulations,
        guaranteeing clean flow conservation (inflow == outflow) at all internal nodes.
        """
        for node in range(self.n):
            if node == self.source or node == self.sink:
                continue

            while self.excess[node] > 1e-9:
                path_edges: List[FlowEdge] = []
                visited: Dict[int, int] = {}
                curr = node

                while curr != self.source and curr not in visited:
                    visited[curr] = len(path_edges)
                    found = False
                    for edge in self.network.adjacency[curr]:
                        rev = self.network.get_residual_edge(edge)
                        if rev.flow > 1e-9:
                            path_edges.append(rev)
                            curr = edge.target
                            found = True
                            break
                    if not found:
                        break

                if curr == self.source:
                    # Path found: s -> ... -> node
                    bottleneck = self.excess[node]
                    for e in path_edges:
                        bottleneck = min(bottleneck, e.flow)
                    for e in path_edges:
                        e.flow -= bottleneck
                        rev = self.network.get_residual_edge(e)
                        rev.flow += bottleneck
                    self.excess[node] -= bottleneck
                    self.excess[self.source] += bottleneck
                elif curr in visited:
                    # Cycle detected: eliminate circulation
                    cycle_start = visited[curr]
                    cycle_edges = path_edges[cycle_start:]
                    bottleneck = min(e.flow for e in cycle_edges)
                    for e in cycle_edges:
                        e.flow -= bottleneck
                        rev = self.network.get_residual_edge(e)
                        rev.flow += bottleneck
                else:
                    break

    def compute_max_flow(self) -> float:
        """Solves the max flow problem and returns the total flow value."""
        # 1. Setup initial heights
        self.height[self.source] = self.n
        self.height[self.sink] = 0
        self.height_count[self.n] = 1
        self.height_count[0] = self.n - 1

        # 2. Saturate all source outgoing edges (initial preflow)
        for edge in self.network.adjacency[self.source]:
            cap = edge.capacity
            if cap > 1e-9:
                edge.flow = cap
                rev = self.network.get_residual_edge(edge)
                rev.flow = -cap
                self.excess[self.source] -= cap
                self.excess[edge.target] += cap

        # 3. Queue up active nodes
        for node in range(self.n):
            if node != self.source and node != self.sink and self.excess[node] > 1e-9:
                self._mark_active(node)

        # 4. Discharge nodes starting from highest active bucket
        while self.max_active_height >= 0:
            bucket = self.active_buckets[self.max_active_height]
            if not bucket:
                self.max_active_height -= 1
                continue

            node = bucket.pop()
            if self.height[node] == self.max_active_height and self.excess[node] > 1e-9:
                self._discharge(node)

            if self._work_counter >= self._global_relabel_frequency:
                self._global_relabel()

        max_flow_val = float(self.excess[self.sink])

        # 5. Return leftover internal excess to source so flow conservation holds
        self._convert_preflow_to_flow()

        return max_flow_val

    def compute_min_cut(self) -> Tuple[Set[int], Set[int]]:
        """Finds minimum s-t cut via reachable nodes from s in residual graph."""
        visited: Set[int] = set()
        queue: deque[int] = deque([self.source])
        visited.add(self.source)

        while queue:
            curr = queue.popleft()
            for edge in self.network.adjacency[curr]:
                if edge.residual_capacity > 1e-9 and edge.target not in visited:
                    visited.add(edge.target)
                    queue.append(edge.target)

        cut_s = visited
        cut_t = set(range(self.n)) - visited
        return cut_s, cut_t

    def get_flow_assignment(self) -> Dict[Tuple[int, int], float]:
        """Returns dict of (u, v) -> flow for all original edges."""
        flows: Dict[Tuple[int, int], float] = {}
        for u in range(self.n):
            for edge in self.network.adjacency[u]:
                if edge.capacity > 0:
                    flows[(edge.source, edge.target)] = edge.flow
        return flows

    def verify_optimality(self, tolerance: float = 1e-7) -> bool:
        """Sanity check: asserts capacity limits and node flow conservation."""
        for u in range(self.n):
            for edge in self.network.adjacency[u]:
                if edge.capacity > 0:
                    if edge.flow < -tolerance or edge.flow > edge.capacity + tolerance:
                        raise FlowConservationError(
                            f"Capacity bounds exceeded on ({edge.source}->{edge.target}): "
                            f"flow={edge.flow}, cap={edge.capacity}"
                        )

        for u in range(self.n):
            if u == self.source or u == self.sink:
                continue

            inflow = 0.0
            outflow = 0.0
            for edge in self.network.adjacency[u]:
                if edge.capacity > 0:
                    outflow += edge.flow
                else:
                    inflow += (-edge.flow)

            if abs(inflow - outflow) > tolerance:
                raise FlowConservationError(
                    f"Conservation failed at node {u}: "
                    f"in={inflow}, out={outflow}, diff={abs(inflow - outflow)}"
                )

        return True
