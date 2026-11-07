# Python Algorithmic Samples

This repository contains two practical algorithmic implementations written from scratch in pure Python (standard library only, zero external dependencies):

1. **A* Search Algorithm (`sample_1_astar/`)**: Heuristic graph pathfinding supporting arbitrary weighted graphs and 2D obstacle grids with diagonal corner-cutting prevention.
2. **O(1) LRU Cache (`sample_2_lru_cache/`)**: High-performance Least Recently Used cache built using a Doubly Linked List, Hash Map, sentinel nodes, and optional TTL expiration.

Both implementations emphasize the idea of **"conceptually clean, algorithmically non-trivial"**—simple to follow and reason about, but requiring careful invariant maintenance, optimal asymptotic complexity, and comprehensive edge-case handling.

---

## Project Structure

```text
new/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── run_all_tests.py                 # Test runner (executes all 20 tests)
├── sample_1_astar/                  # Sample 1: A* Pathfinding
│   ├── __init__.py
│   ├── astar.py                     # Core algorithm & grid graph
│   └── test_astar.py                # 9 unit tests
└── sample_2_lru_cache/              # Sample 2: O(1) LRU Cache
    ├── __init__.py
    ├── lru_cache.py                 # Doubly linked list + hash map
    └── test_lru_cache.py            # 11 unit tests
```

---

## Sample 1: A* Pathfinding (`sample_1_astar/`)

### Why It's "Easy but Hard"
The core concept is intuitive: like Dijkstra's algorithm, but guided by a heuristic function estimating remaining distance ($f(n) = g(n) + h(n)$).

However, writing a robust, production-grade A* requires addressing several tricky details:
- **Generic Graph Abstraction:** Decoupled through a `Graph[T]` protocol so it works over abstract node graphs, road networks, or 2D grids.
- **Priority Queue Tie-breaking:** Using an internal incrementing counter so `heapq` never fails when comparing nodes with identical $f$-scores.
- **Diagonal Movement & Corner Pinching:** On 8-directional grids, diagonal movement must prevent "corner-cutting" through adjacent impassable obstacles.
- **Heuristic Compatibility:** Built-in admissible heuristics (Manhattan for 4-way movement, Euclidean, and Octile for 8-way movement).

### Quick Example
```python
from sample_1_astar import GridGraph, astar_search, manhattan_distance

# Create a 10x10 grid with obstacles
obstacles = {(2, 1), (2, 2), (2, 3)}
grid = GridGraph(width=10, height=10, obstacles=obstacles)

# Find optimal path from (0, 2) to (5, 2)
result = astar_search(
    graph=grid,
    start=(0, 2),
    goal=(5, 2),
    heuristic=manhattan_distance,
)

if result:
    print(f"Path found! Cost: {result.cost}, Explored: {result.nodes_explored} nodes")
    print(f"Waypoints: {result.path}")
```

---

## Sample 2: O(1) LRU Cache (`sample_2_lru_cache/`)

### Why It's "Easy but Hard"
Everyone understands what an LRU cache does: when capacity is reached, drop the item that hasn't been accessed for the longest time.

Implementing it with true **$O(1)$ time complexity** for both `get` and `put` from scratch without relying on Python's built-in `collections.OrderedDict` requires building a custom Doubly Linked List and Hash Map:
- **Sentinel Nodes:** Uses dummy `head` and `tail` sentinels to eliminate null-pointer boundary checks during node insertion, deletion, and relocation.
- **Exact Pointer Relocation:** Relocates nodes to the MRU (Most Recently Used) position in $O(1)$ time upon access.
- **Non-Mutating Peeking:** Provides a `.peek(key)` method to read values without refreshing the LRU eviction priority.
- **Optional TTL Expiration:** Supports per-item or default time-to-live expiration based on Python's monotonic clock (`time.monotonic()`).

### Quick Example
```python
from sample_2_lru_cache import LRUCache

# Initialize cache with max capacity of 2 items
cache = LRUCache(capacity=2)

cache.put("user_1", {"name": "Alice"})
cache.put("user_2", {"name": "Bob"})

# Access user_1 (marks it as Most Recently Used)
print(cache.get("user_1"))

# Insert user_3 (capacity exceeded -> evicts least recently used 'user_2')
cache.put("user_3", {"name": "Charlie"})

print(cache.get("user_2"))  # None (evicted)
print("user_1" in cache)    # True
```

---

## Running the Tests

All tests run using Python's standard `unittest` library (zero installation required, fully compatible with `pytest`):

```bash
# Run all 20 tests across both samples
python run_all_tests.py

# Or run tests individually
python -m unittest discover -s sample_1_astar
python -m unittest discover -s sample_2_lru_cache
```

The test suites verify:
- **A\* Pathfinding:** Straight paths, labyrinth/obstacle avoidance, diagonal movement with corner-pinch prevention, unreachable targets, custom non-grid weighted graphs, and heuristic pruning efficiency vs Dijkstra.
- **LRU Cache:** Eviction order, access refreshes, overwriting existing keys, single-capacity edge cases, `peek()` vs `get()`, deletions, clearing, iteration order (MRU to LRU), and TTL expiration.
