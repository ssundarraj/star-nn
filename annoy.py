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
    # TODO: implement this
    pass


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
    # TODO: implement this
    pass


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
        # TODO: implement this
        pass

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
        # TODO: implement this
        pass

    # ============================================================
    # STEP 5: Build the forest
    # ============================================================
    def build(self, training_data: list[tuple[list[float], str]]) -> None:
        """Build n_trees independent random trees over the full dataset.

        Each tree sees all points but makes different random splits.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 6: Query one tree
    # ============================================================
    def _query_tree(self, node: Node, query: list[float]) -> list[int]:
        """Traverse a single tree to find the leaf containing the query.

        At each internal node, test which side of the hyperplane the
        query falls on: n · q - d <= 0 → go left, else → go right.

        Returns the indices stored in the reached leaf.
        """
        # TODO: implement this
        pass

    # ============================================================
    # STEP 7: Query the forest
    # ============================================================
    def query(self, query: list[float], k: int) -> list[int]:
        """Find approximate k nearest neighbors.

        1. Collect candidate indices from all trees (union/deduplicate).
        2. Rank candidates by true distance to query.
        3. Return the top k.
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
