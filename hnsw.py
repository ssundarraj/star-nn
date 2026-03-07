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
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._data: list[tuple[list[float], str]] = [] # list[(vector, label)]
        self._layers: list[dict[int, list[int]]] = [] # list of dict: point -> adj points
        self._entry_point: int | None = None
        self._max_level: int = 0

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
        return int(-math.log(max(random.random(), 1e-10)) / math.log(self.M))

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
        candidates : list[tuple[float, int]] = []
        results : list[tuple[float, int]] = []
        visited: set[int] = set()
        for p in entry_points:
            d = euclidean_distance(query, self._data[p][0])
            heapq.heappush(candidates, (d, p))
            heapq.heappush(results, (-d, p))
            visited.add(p)

        while len(candidates):
            c = heapq.heappop(candidates)
            c_dist = c[0]
            f = results[0]
            f_dist = -f[0] # we push -dist for maxheap
            if c_dist > f_dist and len(results) >= ef:
                break
            adj_nodes = self._layers[layer][c[1]]
            for adj_node in adj_nodes:
                if adj_node in visited:
                    continue
                visited.add(adj_node)
                adj_node_dist = euclidean_distance(query, self._data[adj_node][0])
                if len(results) < ef or adj_node_dist < f_dist:
                    heapq.heappush(candidates, (adj_node_dist, adj_node))
                    heapq.heappush(results, (-adj_node_dist, adj_node))
                if len(results) > ef:
                    heapq.heappop(results)
        return sorted([(-nd, p) for nd, p in results])

    # ============================================================
    # STEP 4: Select neighbors
    # ============================================================
    def _select_neighbors(
        self,
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
        return [p for (_d, p) in sorted(candidates)][:M]

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
        node_layer = self._random_level()
        self._data.append((features, label))
        node_idx = len(self._data) - 1

        while len(self._layers) < node_layer + 1:
              self._layers.append({})
        for layer_index in range(node_layer + 1):
            self._layers[layer_index][node_idx] = []

        if self._entry_point is None:
            self._entry_point = node_idx
            self._max_level = node_layer
            return 

        cur_layer = self._max_level 
        candidates = [(0.0, self._entry_point)] # arbitrary dist
        
        while cur_layer >= 0:
            ef = 1 if cur_layer > node_layer else self.ef_construction
            candidate_idxs = [c for _d, c in candidates]
            candidates = self._search_layer(
                features, candidate_idxs, ef=ef, layer=cur_layer)
            neighbors = self._select_neighbors(candidates, self.M)
            if node_layer >= cur_layer:
                for n in neighbors:
                    self._layers[cur_layer][node_idx].append(n)
                    self._layers[cur_layer][n].append(node_idx)
                    if len(self._layers[cur_layer][n]) > self.M:
                        n_features = self._data[n][0]
                        scored = sorted(
                            self._layers[cur_layer][n],
                            key=lambda nb: euclidean_distance(
                                n_features, self._data[nb][0]))
                        self._layers[cur_layer][n] = scored[:self.M]
            cur_layer -= 1


        if node_layer > self._max_level:
            self._entry_point = node_idx
            self._max_level = node_layer
        pass

    # ============================================================
    # STEP 6: Build the index
    # ============================================================
    def build(self, training_data: list[tuple[list[float], str]]) -> None:
        """Build the HNSW index by inserting all points one by one.

        Args:
            training_data: The dataset as (features, label) tuples.
        """
        for features, label in training_data:
            self.insert(features, label)

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
        cur_layer = self._max_level
        if self._entry_point is None:
            return []
        entry_point : int = self._entry_point
        neighbors : list[int] = []
        while cur_layer >= 0:
            ef = 1 if cur_layer > 0 else max(self.ef_search, k)
            candidates = self._search_layer(
                query, [entry_point], ef=ef, layer=cur_layer)
            neighbors = self._select_neighbors(candidates, k)
            entry_point = neighbors[0]
            cur_layer -= 1

        return neighbors




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
        correct = sum(1 for features, label in test_data if classify(self.query(features, k), self._data) == label)
        return correct / len(test_data)
