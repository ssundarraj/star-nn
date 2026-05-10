# annoy.py — Approximate Nearest Neighbors Oh Yeah
#
# Annoy builds a forest of random binary trees. Each tree recursively
# splits the dataset by picking two random points and creating a
# hyperplane (perpendicular bisector) between them. At query time,
# traverse each tree to a leaf, union the candidates, then brute-force
# rank them to find the true k nearest.
#
# Run tests with: python3 test_annoy.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import random

from knn import euclidean_distance, classify


# A node in the random binary tree.
# Leaf nodes store point indices. Internal nodes store a hyperplane
# (normal vector + offset) and left/right children.
@dataclass
class Node:
    is_leaf: bool
    indices: list[int] = field(default_factory=list)
    left: Optional[Node] = None
    right: Optional[Node] = None
    normal: list[float] = field(default_factory=list)
    offset: float = 0.0


# ============================================================
# STEP 1: Dot product
# ============================================================
def _dot(a: list[float], b: list[float]) -> float:
    """Return the dot product of two equal-length vectors.

    a · b = Σ(a_i * b_i)
    """
    return sum([ai * bi for ai, bi in zip(a, b)])


# ============================================================
# STEP 2: Split hyperplane
# ============================================================
def _make_split_hyperplane(
    point_a: list[float],
    point_b: list[float],
) -> tuple[list[float], float]:
    """Compute the hyperplane that is the perpendicular bisector between
    point_a and point_b.

    A hyperplane in R^n is defined by: { x | n · x = d }
    where n is the normal vector and d is the offset.

    For the perpendicular bisector:
      - The normal n = b - a (points from a toward b)
      - The plane passes through the midpoint m = (a + b) / 2
      - The offset d = n · m

    Side test: a point x is on the LEFT if n · x - d <= 0

    Returns:
        (normal, offset)
    """
    normal = [bi - ai for ai, bi in zip(point_a, point_b)]
    midpoint  = [(ai + bi)/2 for ai, bi in zip(point_a, point_b)]
    offset = _dot(normal, midpoint)
    return  normal, offset


class AnnoyIndex:
    """An approximate nearest neighbor index using a forest of random
    binary trees.

    Usage:
        index = AnnoyIndex(n_trees=10, max_leaf_size=10)
        index.build(training_data)
        neighbors = index.query(query_point, k=5)

    Parameters:
        n_trees:       Number of random trees to build. More trees = better
                       recall, slower build and query.
        max_leaf_size: Stop splitting when a node has this many or fewer
                       points.
    """

    # ============================================================
    # STEP 3: Constructor
    # ============================================================
    def __init__(self, n_trees: int = 10, max_leaf_size: int = 10) -> None:
        """Store hyperparameters and initialize an empty forest."""
        self.n_trees = n_trees
        self.max_leaf_size = max_leaf_size
        self._forest: list[Node] = []
        self._training_data: list[tuple[list[float], str]] = []

    def _split_over_hyperplane(
        self,
        indices: list[int],
        hyperplane: tuple[list[float], float]
    ) -> tuple[list[int], list[int]]:
        normal, offset = hyperplane
        left: list[int] = []
        right: list[int] = []
        for idx in indices:
            point = self._training_data[idx][0]
            is_left = (_dot(normal, point) - offset) <= 0
            if is_left:
                left.append(idx)
            else:
                right.append(idx)
        return left, right


    # ============================================================
    # STEP 4: Build one tree
    # ============================================================
    def _build_tree(self, indices: list[int]) -> Node:
        """Recursively build one random binary tree over the given point
        indices.

        Base case:
            |indices| <= max_leaf_size → return a leaf.

        Recursive case:
            1. Sample two distinct points p, q from the subset.
            2. Compute the perpendicular bisector hyperplane between p and q.
            3. Partition all indices by which side of the hyperplane they
               fall on: n · x_i - d <= 0 → left, otherwise → right.
            4. If either partition is empty, return a leaf (degenerate split).
            5. Recurse on each partition.

        Returns:
            Root Node of this subtree.
        """
        if (len(indices) <= self.max_leaf_size):
            return Node(is_leaf=True, indices = indices)

        idx_a, idx_b = random.sample(indices, 2)
        point_a, point_b = self._training_data[idx_a][0], self._training_data[idx_b][0]
        hyperplane = _make_split_hyperplane(point_a, point_b)
        left_idxs, right_idxs = self._split_over_hyperplane(indices, hyperplane)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            # degenerate split
            return Node(is_leaf=True, indices = indices)
        left = self._build_tree(left_idxs)
        right = self._build_tree(right_idxs)
        normal, offset = hyperplane
        return Node(is_leaf=False, left=left, right=right, normal=normal, offset=offset)

    # ============================================================
    # STEP 5: Build the forest
    # ============================================================
    def build(self, training_data: list[tuple[list[float], str]]) -> None:
        """Build n_trees independent random trees over the full dataset.

        Each tree sees all points but makes different random splits.
        """
        self._training_data = training_data

        all_idxs = list(range(len(training_data)))
        for _ in range(self.n_trees):
            self._forest.append(self._build_tree(all_idxs))

    # ============================================================
    # STEP 6: Query one tree
    # ============================================================
    def _query_tree(self, node: Node, query: list[float]) -> list[int]:
        """Traverse a single tree to find the leaf containing the query.

        At each internal node, test which side of the hyperplane the
        query falls on: n · q - d <= 0 → go left, else → go right.

        Returns the indices stored in the reached leaf.
        """
        if (node.is_leaf):
            return node.indices
        assert node.left # Node dataclass needs better typing?
        assert node.right
        is_left = (_dot(node.normal, query) - node.offset) <= 0
        if (is_left):
            return self._query_tree(node.left, query)
        else:
            return self._query_tree(node.right, query)
        

    # ============================================================
    # STEP 7: Query the forest
    # ============================================================
    def query(self, query: list[float], k: int) -> list[int]:
        """Find approximate k nearest neighbors.

        1. Collect candidate indices from all trees (union/deduplicate).
        2. Rank candidates by true distance to query.
        3. Return the top k.
        """
        candidate_set: set[int] = set()
        for tree in self._forest:
            candidate_set.update(self._query_tree(tree, query))
        ranked = sorted(candidate_set, key=lambda i: euclidean_distance(self._training_data[i][0], query))
        return ranked[:k]



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
        correct = sum(1 for features, label in test_data if classify(self.query(features, k), self._training_data) == label)
        return correct / len(test_data)
