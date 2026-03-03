# test_knn.py — Tests for your KNN implementation
#
# Run with: python test_knn.py
#
# Work through the steps in order — once all tests pass for a step,
# move on to the next.

from knn import (
    euclidean_distance,
    manhattan_distance,
    find_k_nearest,
    classify,
    predict,
)

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


# --- Step 1a: Euclidean Distance ---
print("\n--- Step 1a: Euclidean Distance ---")

check(euclidean_distance([0, 0], [0, 0]) == 0.0, "origin to origin")
check(euclidean_distance([0, 0], [3, 4]) == 5.0, "3-4-5 triangle")
check(euclidean_distance([1], [4]) == 3.0, "1D distance")

d1 = euclidean_distance([1, 2, 3], [4, 5, 6])
d2 = euclidean_distance([4, 5, 6], [1, 2, 3])
check(abs(d1 - d2) < 1e-9, "symmetry")

# --- Step 1b: Manhattan Distance ---
print("\n--- Step 1b: Manhattan Distance ---")

check(manhattan_distance([0, 0], [0, 0]) == 0.0, "origin to origin")
check(manhattan_distance([0, 0], [3, 4]) == 7.0, "3+4=7")
check(manhattan_distance([1, 2], [4, 6]) == 7.0, "3+4=7 shifted")

# --- Step 2: Find K Nearest ---
print("\n--- Step 2: Find K Nearest ---")

data = [
    ([0, 0], "A"),
    ([1, 0], "A"),
    ([10, 10], "B"),
    ([0.5, 0.5], "A"),
    ([9, 9], "B"),
]

neighbors = find_k_nearest([0, 0], data, 3)
check(len(neighbors) == 3, "returns 3 neighbors")
check(set(neighbors) == {0, 1, 3}, "correct nearest neighbors to origin")

# --- Step 3: Classify ---
print("\n--- Step 3: Classify ---")

data2 = [
    ([0, 0], "cat"),
    ([1, 0], "cat"),
    ([2, 0], "dog"),
    ([3, 0], "cat"),
]

check(classify([0, 1, 2], data2) == "cat", "majority vote cat")
check(classify([2], data2) == "dog", "single neighbor dog")

# --- Step 4: Predict ---
print("\n--- Step 4: Predict ---")

data3 = [
    ([0, 0], "A"),
    ([1, 1], "A"),
    ([2, 2], "A"),
    ([10, 10], "B"),
    ([11, 11], "B"),
    ([12, 12], "B"),
]

check(predict([0.5, 0.5], data3, 3) == "A", "predict near A cluster")
check(predict([10.5, 10.5], data3, 3) == "B", "predict near B cluster")

# --- Summary ---
print(f"\n=== Results: {passed} passed, {failed} failed ===")
exit(1 if failed > 0 else 0)
