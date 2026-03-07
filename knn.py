from collections import defaultdict
import heapq
import math
from functools import reduce

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
# Uses a max-heap of size k. We push (-dist, idx) so the farthest
# candidate is always at the top. When a closer point is found,
# pop the farthest and push the new one. O(n log k) total.
def find_k_nearest(
    query: list[float],
    training_data: list[tuple[list[float], str]],
    k: int,
) -> list[int]:
    heap: list[tuple[float, int]] = []  # max-heap via negated distances
    for i, (point, _label) in enumerate(training_data):
        dist = euclidean_distance(point, query)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, i))
        elif dist < -heap[0][0]:
            heapq.heapreplace(heap, (-dist, i))

    return [i for _, i in sorted(heap, reverse=True)]

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
