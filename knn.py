from collections import defaultdict
import math
from typing import NamedTuple
from functools import reduce

class PointAndDistance(NamedTuple):
    idx: int
    distance: float

def euclidian_dist(
    x: list[float],
    y: list[float]
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

# returns indexes of k nearest
def find_k_nearest(
    query: list[float],
    training_data: list[tuple[list[float], str]],
    k: int,
) -> list[int]:
    k_nearest: list[PointAndDistance] = []
    for i, (point, _label) in enumerate(training_data):
        dist = euclidian_dist(point, query)

        if len(k_nearest) < k:
            k_nearest.append(PointAndDistance(i, dist))
        elif k_nearest[-1].distance > dist:
            k_nearest[-1] = PointAndDistance(i, dist)

        # TODO: insertion sort
        k_nearest = sorted(k_nearest) 

    return [point.idx for point in k_nearest]

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
