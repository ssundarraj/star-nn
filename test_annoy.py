# test_annoy.py — Tests for your Annoy implementation
#
# Run with: python3 test_annoy.py
#
# Work through the steps in order — once all tests pass for a step,
# move on to the next.

from annoy import _dot, _make_split_hyperplane, AnnoyIndex, Node

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


# --- Step 1: Dot Product ---
print("\n--- Step 1: Dot Product ---")

check(_dot([1, 0], [1, 0]) == 1.0, "unit vectors")
check(_dot([1, 2], [3, 4]) == 11.0, "general case")
check(_dot([0, 0], [5, 5]) == 0.0, "zero vector")

# --- Step 2: Split Hyperplane ---
print("\n--- Step 2: Split Hyperplane ---")

try:
    normal, offset = _make_split_hyperplane([0.0, 0.0], [2.0, 0.0])
    midpoint = [1.0, 0.0]
    check(abs(_dot(normal, midpoint) - offset) < 1e-9, "midpoint lies on hyperplane")
    check(_dot(normal, [0.0, 0.0]) - offset <= 0, "point_a is on left side")
    check(_dot(normal, [2.0, 0.0]) - offset >= 0, "point_b is on right side")
except TypeError:
    print("  SKIP: _make_split_hyperplane not yet implemented")

# --- Steps 3-5: Build ---
print("\n--- Steps 3-5: Build ---")

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
    index = AnnoyIndex(n_trees=5, max_leaf_size=3)
    index.build(data)
    check(len(index._forest) == 5, "forest has 5 trees")
    check(all(isinstance(t, Node) for t in index._forest), "all roots are Nodes")
except (TypeError, AttributeError):
    print("  SKIP: build not yet implemented")
    index = None

# --- Step 6: Query Tree ---
print("\n--- Step 6: Query Tree ---")

try:
    if index and index._forest:
        root = index._forest[0]
        candidates = index._query_tree(root, [0.05, 0.05])
        check(isinstance(candidates, list), "query_tree returns a list")
        check(len(candidates) > 0, "at least one candidate")
        check(all(0 <= i < len(data) for i in candidates), "all indices valid")
    else:
        print("  SKIP: need build to work first")
except TypeError:
    print("  SKIP: _query_tree not yet implemented")

# --- Step 7: Query Forest ---
print("\n--- Step 7: Query Forest ---")

try:
    if index and index._forest:
        neighbors_a = index.query([0.05, 0.05], k=3)
        check(len(neighbors_a) == 3, "returns k neighbors")
        check(all(i < 5 for i in neighbors_a), "neighbors near A are from A cluster")

        neighbors_b = index.query([10.05, 10.05], k=3)
        check(all(i >= 5 for i in neighbors_b), "neighbors near B are from B cluster")
    else:
        print("  SKIP: need build to work first")
except TypeError:
    print("  SKIP: query not yet implemented")

# --- Step 8: Evaluate ---
print("\n--- Step 8: Evaluate ---")

try:
    from csv_loader import load_csv
    iris = load_csv("data/iris.csv")
    if iris:
        split = len(iris) * 80 // 100
        train, test = iris[:split], iris[split:]
        idx = AnnoyIndex(n_trees=10, max_leaf_size=10)
        idx.build(train)
        acc = idx.evaluate(test, k=5)
        check(acc > 0.7, f"Annoy accuracy on Iris > 70% (got {acc:.1%})")
    else:
        print("  SKIP: load_csv not yet implemented")
except (ImportError, TypeError):
    print("  SKIP: csv_loader or annoy not fully implemented")

# --- Summary ---
print(f"\n=== Results: {passed} passed, {failed} failed ===")
exit(1 if failed > 0 else 0)
