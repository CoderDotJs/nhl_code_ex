# Python Algorithmic Samples

This repository contains two algorithmic code samples implemented from scratch in pure Python (standard library only, no external dependencies). I chose these two problems because they show different areas of algorithm engineering: network flow optimization and multi-dimensional spatial indexing.

Both samples include comprehensive test suites with property checks, textbook benchmarks, and randomized stress tests.

---

## Project Structure

```text
new/
├── README.md
├── run_all_tests.py                 # Test runner (runs all 22 tests)
├── sample_1_push_relabel/           # Sample 1: Push-Relabel Max Flow
│   ├── push_relabel.py              # Core implementation
│   ├── test_push_relabel.py         # Unit tests & invariants
│   └── __init__.py
└── sample_2_kd_tree/                # Sample 2: Spatial k-d Tree
    ├── kd_tree.py                   # Core implementation
    ├── test_kd_tree.py              # Unit tests & 5D fuzzing
    └── __init__.py
```

---

## Sample 1: Push-Relabel Maximum Flow (`sample_1_push_relabel/`)

### Why Push-Relabel?
Standard augmenting path algorithms like Edmonds-Karp run in $O(V E^2)$, which gets slow on dense graphs with large capacities. The Goldberg-Tarjan push-relabel approach operates locally by pushing excess flow across neighboring vertices and lifting vertex heights.

With **highest-label selection**, the theoretical runtime drops to $O(V^2 \sqrt{E})$, and practical performance is even faster when paired with heuristics.

### Key Implementation Details
- **Highest-label active node buckets:** Nodes with excess flow are kept in bucket lists grouped by height. Always discharging the highest active vertex first keeps the number of non-saturating pushes low.
- **Gap relabeling heuristic:** If an intermediate height level becomes completely empty, all vertices above that height (up to $n$) can no longer reach the sink in the residual graph. The solver immediately raises their heights to $n + 1$ to avoid wasted work.
- **Global backward BFS relabeling:** Every $4n$ work units, the solver runs a backward BFS from the sink in the residual graph to reset heights to exact shortest-path distances.
- **Preflow-to-flow conversion:** After Phase 1 calculates the max flow to the sink, leftover excess at internal nodes is returned back along flow paths to the source (cancelling any internal flow cycles). This guarantees exact flow conservation ($\sum f_{in} = \sum f_{out}$) at every non-terminal node.
- **Min-cut extraction & verification:** Extracts the $(S, T)$ partition using reachability in the residual network and verifies that cut capacity matches the flow value.

### Quick Example
```python
from sample_1_push_relabel import FlowNetwork, PushRelabelSolver

# Setup a small network
net = FlowNetwork(num_nodes=6)
s, v1, v2, v3, v4, t = 0, 1, 2, 3, 4, 5

net.add_edge(s, v1, 16.0)
net.add_edge(s, v2, 13.0)
net.add_edge(v1, v3, 12.0)
net.add_edge(v2, v1, 4.0)
net.add_edge(v2, v4, 14.0)
net.add_edge(v3, v2, 9.0)
net.add_edge(v3, t, 20.0)
net.add_edge(v4, v3, 7.0)
net.add_edge(v4, t, 4.0)

solver = PushRelabelSolver(net, source=s, sink=t)
max_flow = solver.compute_max_flow()
print(f"Max flow: {max_flow}")  # 23.0

cut_s, cut_t = solver.compute_min_cut()
print(f"Cut partition: S={cut_s}, T={cut_t}")

# Verify conservation and capacity constraints
assert solver.verify_optimality()
```

---

## Sample 2: Multi-Dimensional k-d Tree (`sample_2_kd_tree/`)

### Overview
A balanced $k$-d tree implementation for spatial indexing and nearest-neighbor search in arbitrary dimensions ($\mathbb{R}^k$).

### Key Implementation Details
- **Variance / max-spread axis selection:** Rather than naively cycling through dimensions ($0, 1, \dots, k-1$), each split chooses the coordinate axis with the largest spread ($\max - \min$). This avoids long, narrow bounding boxes on anisotropic or correlated datasets.
- **Branch-and-bound pruning:** Each node keeps a tight bounding box (`HyperRectangle`). During $k$-NN search, we project the query point onto candidate bounding boxes. If the minimum distance to a box cannot beat the current $k$-th best neighbor, the entire subtree is pruned.
- **Metric protocol:** Distance calculations are decoupled through a `Metric` protocol. Included out of the box:
  - Euclidean ($L_2$)
  - Manhattan ($L_1$)
  - Chebyshev ($L_\infty$)
  - Minkowski ($L_p$)
- **Range & radius queries:**
  - `range_search(box)`: returns all points inside an axis-aligned bounding box.
  - `radius_search(center, radius)`: returns all points within a hypersphere.
- **Dynamic insertion & rebalancing:** Supports dynamic point inserts. If a subtree becomes unbalanced (one branch holds more than 70% of the nodes), that subtree is rebuilt on the fly around the median.

### Quick Example
```python
from sample_2_kd_tree import KDPoint, KDTree, HyperRectangle

points = [
    KDPoint((2.0, 3.0), payload="A"),
    KDPoint((5.0, 4.0), payload="B"),
    KDPoint((9.0, 6.0), payload="C"),
    KDPoint((4.0, 7.0), payload="D"),
]
tree = KDTree.build(points)

# 1. Nearest neighbors
neighbors = tree.k_nearest_neighbors(query=(2.1, 3.1), k=2)
for match in neighbors:
    print(f"Point {match.point.payload} at dist {match.distance:.3f}")

# 2. Bounding box range search
box = HyperRectangle(min_bounds=(1.0, 2.0), max_bounds=(6.0, 5.0))
matches = tree.range_search(box)
print(f"Points in box: {[p.payload for p in matches]}")
```

---

## Running the Tests

All tests are written using Python's built-in `unittest` module, so no packages need to be installed.

```bash
# Run all 22 tests across both samples
python new/run_all_tests.py

# Or run tests individually
python -m unittest discover -s new/sample_1_push_relabel
python -m unittest discover -s new/sample_2_kd_tree
```

The test suites verify:
- Edge cases (empty trees, disconnected components, antiparallel edges, invalid inputs)
- Classic benchmarks (CLRS Chapter 26 flow network, diamond topology)
- Flow conservation and max-flow min-cut duality
- 5D randomized fuzzing comparing $k$-NN results directly against brute-force linear searches
