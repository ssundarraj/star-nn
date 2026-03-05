# hnsw.py — Hierarchical Navigable Small World
#
# HNSW is a multi-layer proximity graph. Layer 0 contains all points,
# higher layers contain exponentially fewer points acting as "express
# lanes." Insertion greedily navigates from the top layer down, connecting
# the new point to its nearest neighbors at each layer. Query does the
# same descent, finishing with a beam search at layer 0.
#
# Run tests with: python3 test_hnsw.py

from __future__ import annotations
import math
import random
import heapq

from knn import euclidean_distance, classify


class HNSWIndex:
    """An approximate nearest neighbor index using a hierarchical
    navigable small world graph.

    Usage:
        index = HNSWIndex(M=16, ef_construction=200, ef_search=50)
        index.build(training_data)
        neighbors = index.query(query_point, k=5)

    Parameters:
        M:                Max connections per node per layer.
        ef_construction:  Beam width during insertion (higher = better
                          graph quality, slower build).
        ef_search:        Beam width during query (higher = better
                          recall, slower query).
    """

    # ============================================================
    # STEP 1: Constructor
    # ============================================================
    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50) -> None:
        """Initialize an empty HNSW index.

        Set up:
            - Store M, ef_construction, ef_search
            - self._data: list of (features, label) tuples (populated by insert)
            - self._layers: list of dicts, where self._layers[L][i] is the
              neighbor list for node i at layer L
            - self._entry_point: index of the entry node (None until first insert)
            - self._max_level: current highest layer (0 until points are added)
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 2: Random level assignment
    # ============================================================
    def _random_level(self) -> int:
        """Return a random level for a new node.

        Each node exists at layers 0 through its assigned level.
        The probability of being assigned level L is:
            P(L) = (1/M)^L * (1 - 1/M)

        In practice: draw a uniform random number r in [0, 1),
        return floor(-ln(r) * (1 / ln(M))).

        This gives an exponential distribution — most nodes get level 0,
        few get level 1, very few get level 2, etc.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 3: Search within one layer (beam search)
    # ============================================================
    def _search_layer(
        self,
        query: list[float],
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Beam search within a single layer of the graph.

        This is the core algorithm of HNSW.

        Maintain two structures:
            - candidates: a min-heap of (distance, node_id) — nodes still
              to explore, closest first
            - results: a max-heap of (-distance, node_id) — best ef results
              found so far, farthest first (negated for max-heap behavior)

        Algorithm:
            1. Initialize both heaps with the entry points.
            2. Track visited nodes in a set.
            3. While candidates is not empty:
               a. Pop the closest candidate c.
               b. Pop (peek) the farthest result f.
               c. If dist(c) > dist(f) and we have ef results, stop
                  (no candidate can improve our results).
               d. For each neighbor n of c in this layer:
                  - Skip if already visited.
                  - Compute dist(query, n).
                  - If results has fewer than ef entries, or dist(n) < dist(f):
                    add n to both candidates and results.
                  - If results exceeds ef entries, pop the farthest.
            4. Return results as a list of (distance, node_id), closest first.

        Args:
            query:        The query feature vector.
            entry_points: Starting node indices for the search.
            ef:           Beam width (number of results to track).
            layer:        Which layer to search in.

        Returns:
            List of (distance, node_id) tuples, sorted closest first.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 4: Select neighbors
    # ============================================================
    def _select_neighbors(
        self,
        query: list[float],
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """Select the M closest candidates to the query point.

        Sort candidates by distance, return the first M node indices.

        Args:
            query:      The query feature vector.
            candidates: List of (distance, node_id) tuples.
            M:          Max number of neighbors to select.

        Returns:
            List of at most M node indices.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 5: Insert a single point
    # ============================================================
    def insert(self, features: list[float], label: str) -> None:
        """Insert one data point into the index.

        Algorithm:
            1. Append (features, label) to self._data. Its index is
               node_id = len(self._data) - 1.
            2. Assign a random level L via self._random_level().
            3. Ensure self._layers has at least L+1 layers (extend if needed).
               Add node_id with an empty neighbor list at each layer 0..L.
            4. If this is the first point, set it as entry_point and return.
            5. Navigate from top layer down to L+1:
               At each layer, use _search_layer with ef=1 to find the
               single closest node. Use that as the entry point for the
               next layer down. This is the "greedy descent" phase.
            6. At layers L down to 0:
               Use _search_layer with ef=ef_construction to find candidates.
               Use _select_neighbors to pick M neighbors.
               Connect node_id to each neighbor (bidirectional — add edges
               both ways). If a neighbor exceeds M connections, trim to
               keep only its M closest.
            7. If L > self._max_level, update entry_point and max_level.

        Args:
            features: The point's feature vector.
            label:    The point's class label.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 6: Build the index
    # ============================================================
    def build(self, training_data: list[tuple[list[float], str]]) -> None:
        """Build the HNSW index by inserting all points one by one.

        Args:
            training_data: The dataset as (features, label) tuples.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 7: Query
    # ============================================================
    def query(self, query: list[float], k: int) -> list[int]:
        """Find approximate k nearest neighbors.

        Algorithm:
            1. Start at self._entry_point.
            2. Greedy descent from self._max_level down to layer 1:
               at each layer, _search_layer with ef=1, use the closest
               result as entry point for the next layer.
            3. At layer 0, _search_layer with ef=max(ef_search, k).
            4. Return the k closest node indices.

        Args:
            query: The query feature vector.
            k:     Number of neighbors to return.

        Returns:
            List of at most k indices into self._data.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 8: Evaluate
    # ============================================================
    def evaluate(
        self,
        test_data: list[tuple[list[float], str]],
        k: int,
    ) -> float:
        """Classification accuracy: for each test point, query the index,
        majority-vote the neighbors' labels, compare to true label.

        Returns accuracy in [0.0, 1.0].
        """
        # TODO: implement this
        pass
