# main.py — Entry point for your KNN classifier
#
# Once you've implemented the functions in knn.py and csv_loader.py,
# this program will:
#   1. Load the Iris dataset from a CSV file.
#   2. Split it into training and test sets.
#   3. Run KNN classification and print accuracy.
#
# Run with: python main.py

import time
from knn import evaluate
from csv_loader import load_csv
from annoy import AnnoyIndex

data = load_csv("data/iris.csv")

if not data:
    print("Failed to load data. Check the file path.")
    exit(1)

print(f"Loaded {len(data)} data points.")

# Simple train/test split: first 80% train, last 20% test
split = len(data) * 80 // 100
train = data[:split]
test = data[split:]

print(f"Training set: {len(train)} points")
print(f"Test set:     {len(test)} points")

# Try different values of K
for k in [1, 3, 5, 7]:
    accuracy = evaluate(test, train, k)
    print(f"K={k}  accuracy={accuracy * 100:.1f}%")

# --- Annoy vs Brute-Force KNN ---
print("\n--- Annoy vs Brute-Force KNN ---")

for n_trees in [5, 10, 20]:
    t0 = time.perf_counter()
    index = AnnoyIndex(n_trees=n_trees, max_leaf_size=10)
    index.build(train)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = index.evaluate(test, k=5)
    query_time = time.perf_counter() - t0

    print(f"  n_trees={n_trees:2d}  accuracy={acc * 100:.1f}%  "
          f"build={build_time * 1000:.1f}ms  query={query_time * 1000:.1f}ms")

t0 = time.perf_counter()
knn_acc = evaluate(test, train, k=5)
knn_time = time.perf_counter() - t0
print(f"  KNN (brute)  accuracy={knn_acc * 100:.1f}%  "
      f"query={knn_time * 1000:.1f}ms")
