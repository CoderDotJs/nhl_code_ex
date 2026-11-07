"""A* (A-Star) Pathfinding Package."""

from .astar import (
    Graph,
    GridGraph,
    SearchResult,
    astar_search,
    euclidean_distance,
    manhattan_distance,
    octile_distance,
)

__all__ = [
    "Graph",
    "GridGraph",
    "SearchResult",
    "astar_search",
    "euclidean_distance",
    "manhattan_distance",
    "octile_distance",
]

