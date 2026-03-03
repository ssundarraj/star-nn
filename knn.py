from collections import defaultdict
import math
from typing import NamedTuple
from functools import reduce

class PointAndDistance(NamedTuple):
    idx: int
    distance: float

# [claude review] zip silently truncates if lists are different lengths.
# Fine if you trust your data, but worth knowing.
def euclidean_distance(
    x: list[float],
    y: list[float]
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

# [claude review] same zip truncation note as above.
def manhattan_distance(
    x: list[float],
    y: list[float]
) -> float:
    return sum(abs(a - b) for a, b in zip(x, y))

# returns indexes of k nearest
#
# [claude review] Bug: you're sorting the entire list on every iteration.
# k_nearest[-1] is always the farthest of your current candidates, which
# is correct for the comparison -- but you're doing an O(k log k) sort per
# data point, making the whole thing O(n k log k). Not wrong, but wasteful.
#
# Your approach is heading toward an O(n k) selection algorithm (once you
# replace sorted with insertion sort), which is better for large n -- but
# the sort-per-iteration negates that.
#
# Simpler approach: compute all distances, sort once, slice:
#
#   def find_k_nearest(query, training_data, k):
#       distances = [(euclidean_distance(point, query), i)
#                    for i, (point, _) in enumerate(training_data)]
#       distances.sort()
#       return [i for _, i in distances[:k]]
#
# Same O(n log n) but clearer.
def find_k_nearest(
    query: list[float],
    training_data: list[tuple[list[float], str]],
    k: int,
) -> list[int]:
    k_nearest: list[PointAndDistance] = []
    for i, (point, _label) in enumerate(training_data):
        dist = euclidean_distance(point, query)

        if len(k_nearest) < k:
            k_nearest.append(PointAndDistance(i, dist))
        elif k_nearest[-1].distance > dist:
            k_nearest[-1] = PointAndDistance(i, dist)

        # TODO: insertion sort
        k_nearest = sorted(k_nearest)

    return [point.idx for point in k_nearest]

# [claude review] Works but reduce with a mutating accumulator is
# unidiomatic Python. A loop or Counter is more natural:
#
#   from collections import Counter
#   counts = Counter(training_data[i][1] for i in idxs)
#   return counts.most_common(1)[0][0]
def classify(
    idxs: list[int],
    training_data: list[tuple[list[float], str]],
) -> str:
    def count_labels(acc: defaultdict[str, int], idx: int):
        acc[training_data[idx][1]] += 1
        return acc

    freq_map : dict[str, int] = reduce(count_labels, idxs, defaultdict(int))
    max_idx = max(freq_map, key = lambda label: freq_map[label])
    return max_idx


def predict(
    query: list[float],
    training_data: list[tuple[list[float], str]],
    k: int,
) -> str:
    k_nearest_points = find_k_nearest(query, training_data, k)
    return classify(k_nearest_points, training_data)

def evaluate(
    test_data: list[tuple[list[float], str]],
    training_data: list[tuple[list[float], str]],
    k: int,
) -> float:
    correct = sum(1 for features, label in test_data if predict(features, training_data, k) == label)
    return correct / len(test_data)
