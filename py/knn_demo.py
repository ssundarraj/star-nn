# knn.py — K-Nearest Neighbors
#
# KNN is one of the simplest machine learning algorithms:
#   1. Store all training data (features + labels).
#   2. To predict a new point, find the K closest training points.
#   3. Return the most common label among those K neighbors (classification)
#      or the average of their values (regression).
#
# Your job: implement each function marked with TODO.
# Work through them in order — each builds on the previous one.
#
# No external libraries needed — just Python builtins + math.

import math


# ============================================================
# STEP 1: Distance functions
# ============================================================
# These measure how "close" two points are in feature space.
# Both inputs are lists of numbers with the same length.
#
# Euclidean distance = sqrt( sum of (a[i] - b[i])^2 )
# Manhattan distance = sum of |a[i] - b[i]|

def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Return the Euclidean distance between points a and b."""
    # TODO: implement this
    # Hint: use zip(a, b) to loop through pairs, math.sqrt for square root
    pass


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """Return the Manhattan distance between points a and b."""
    # TODO: implement this
    # Hint: use zip(a, b) and abs()
    pass


# ============================================================
# STEP 2: Find K nearest neighbors
# ============================================================
# Given a query point, training data, a value of K, and a distance function,
# return the indices of the K closest training points.
#
# training_data is a list of (features, label) tuples, e.g.:
#   [([5.1, 3.5, 1.4, 0.2], "Iris-setosa"), ...]
#
# Approach:
#   - Compute distance from query to every training point's features.
#   - Sort by distance.
#   - Return the first K indices.

def find_k_nearest(
    query: list[float],
    training_data: list[tuple[list[float], str]],
    k: int,
    distance_fn=euclidean_distance,
) -> list[int]:
    """Return indices of the K nearest training points to query."""
    # TODO: implement this
    # Hint: enumerate() gives you (index, item) pairs
    #       sorted() with a key= argument can sort by distance
    pass


# ============================================================
# STEP 3: Classify — majority vote
# ============================================================
# Given the indices of the K nearest neighbors, look up their labels
# and return the most frequently occurring label.
#
# Approach:
#   - Count occurrences of each label (use a dict).
#   - Return the label with the highest count.

def classify(
    neighbor_indices: list[int],
    training_data: list[tuple[list[float], str]],
) -> str:
    """Return the most common label among the given neighbors."""
    # TODO: implement this
    # Hint: use a dict to count labels, then max() with key= to find the winner
    pass


# ============================================================
# STEP 4: Putting it all together — predict
# ============================================================
# Given a query point, training data, and K, return the predicted label.

def predict(
    query: list[float],
    training_data: list[tuple[list[float], str]],
    k: int,
) -> str:
    """Predict the label for a query point using KNN."""
    # TODO: implement this (should be 2-3 lines using the functions above)
    pass


# ============================================================
# STEP 5 (bonus): Evaluate accuracy
# ============================================================
# Given test data and training data, predict each test point and compute
# the fraction of correct predictions (accuracy from 0.0 to 1.0).

def evaluate(
    test_data: list[tuple[list[float], str]],
    training_data: list[tuple[list[float], str]],
    k: int,
) -> float:
    """Return accuracy (0.0 to 1.0) of KNN on the test data."""
    # TODO: implement this
    pass
