# test_hnsw.py — Tests for your HNSW implementation
#
# Run with: python3 test_hnsw.py
#
# Work through the steps in order — once all tests pass for a step,
# move on to the next.

from hnsw import HNSWIndex

passed = 0
failed = 0


def check(condition, name):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1


# --- Step 1: Constructor ---
print("\n--- Step 1: Constructor ---")

try:
    idx = HNSWIndex(M=4, ef_construction=20, ef_search=10)
    check(idx.M == 4, "M stored")
    check(idx.ef_construction == 20, "ef_construction stored")
    check(idx.ef_search == 10, "ef_search stored")
    check(idx._entry_point is None, "entry_point starts as None")
except (TypeError, AttributeError) as e:
    print(f"  SKIP: constructor not yet implemented ({e})")

# --- Step 2: Random Level ---
print("\n--- Step 2: Random Level ---")

try:
    idx = HNSWIndex(M=4)
    levels = [idx._random_level() for _ in range(1000)]
    check(all(l >= 0 for l in levels), "all levels non-negative")
    check(levels.count(0) > 500, "most nodes get level 0 (>50%)")
    check(any(l >= 1 for l in levels), "some nodes get level >= 1")
    max_level = max(levels)
    check(max_level < 20, f"max level reasonable ({max_level})")
except TypeError as e:
    print(f"  SKIP: _random_level not yet implemented ({e})")

# --- Steps 3-4: Search Layer & Select Neighbors ---
print("\n--- Steps 3-4: Search Layer & Select Neighbors ---")

try:
    idx = HNSWIndex(M=4, ef_construction=20, ef_search=10)
    # Manually build a small 1-layer graph
    idx._data = [
        ([0.0, 0.0], "A"),
        ([1.0, 0.0], "A"),
        ([2.0, 0.0], "A"),
        ([10.0, 10.0], "B"),
        ([11.0, 10.0], "B"),
    ]
    # Layer 0: 0-1-2 connected, 3-4 connected
    idx._layers = [
        {
            0: [1],
            1: [0, 2],
            2: [1],
            3: [4],
            4: [3],
        }
    ]
    idx._entry_point = 0
    idx._max_level = 0

    results = idx._search_layer([0.5, 0.0], [0], ef=3, layer=0)
    check(len(results) > 0, "search_layer returns results")
    check(results[0][1] in [0, 1], "closest to [0.5, 0] is node 0 or 1")

    neighbors = idx._select_neighbors([0.5, 0.0], results, M=2)
    check(len(neighbors) == 2, "select_neighbors returns M neighbors")
    check(set(neighbors) == {0, 1}, "selects the 2 closest nodes")
except TypeError as e:
    print(f"  SKIP: search_layer/select_neighbors not yet implemented ({e})")

# --- Steps 5-6: Insert & Build ---
print("\n--- Steps 5-6: Insert & Build ---")

cluster_a = [
    ([0.0, 0.0], "A"), ([0.1, 0.0], "A"), ([0.0, 0.1], "A"),
    ([0.2, 0.1], "A"), ([0.1, 0.2], "A"),
]
cluster_b = [
    ([10.0, 10.0], "B"), ([10.1, 10.0], "B"), ([10.0, 10.1], "B"),
    ([10.2, 10.1], "B"), ([10.1, 10.2], "B"),
]
data = cluster_a + cluster_b

try:
    idx = HNSWIndex(M=4, ef_construction=20, ef_search=10)
    idx.build(data)
    check(len(idx._data) == 10, "all 10 points inserted")
    check(len(idx._layers) >= 1, "at least 1 layer exists")
    check(len(idx._layers[0]) == 10, "layer 0 has all 10 nodes")
    check(idx._entry_point is not None, "entry point set")
except (TypeError, AttributeError) as e:
    print(f"  SKIP: build not yet implemented ({e})")
    idx = None

# --- Step 7: Query ---
print("\n--- Step 7: Query ---")

try:
    if idx and idx._entry_point is not None:
        neighbors_a = idx.query([0.05, 0.05], k=3)
        check(len(neighbors_a) == 3, "returns k neighbors")
        check(all(i < 5 for i in neighbors_a), "neighbors near A are from A cluster")

        neighbors_b = idx.query([10.05, 10.05], k=3)
        check(all(i >= 5 for i in neighbors_b), "neighbors near B are from B cluster")
    else:
        print("  SKIP: need build to work first")
except TypeError as e:
    print(f"  SKIP: query not yet implemented ({e})")

# --- Step 8: Evaluate ---
print("\n--- Step 8: Evaluate ---")

try:
    from csv_loader import load_csv
    iris = load_csv("data/iris.csv")
    if iris:
        split = len(iris) * 80 // 100
        train, test = iris[:split], iris[split:]
        idx = HNSWIndex(M=16, ef_construction=200, ef_search=50)
        idx.build(train)
        acc = idx.evaluate(test, k=5)
        check(acc > 0.7, f"HNSW accuracy on Iris > 70% (got {acc:.1%})")
    else:
        print("  SKIP: load_csv not yet implemented")
except (ImportError, TypeError) as e:
    print(f"  SKIP: not fully implemented ({e})")

# --- Summary ---
print(f"\n=== Results: {passed} passed, {failed} failed ===")
exit(1 if failed > 0 else 0)
