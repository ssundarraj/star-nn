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
from mnist_loader import load_mnist
from annoy import AnnoyIndex
from hnsw import HNSWIndex

# =============================================================
# Iris Benchmark (120 train, 30 test, 4 dims)
# =============================================================
print("=" * 60)
print("Iris Benchmark (120 train, 30 test, 4 dims)")
print("=" * 60)

data = load_csv("data/iris.csv")

if not data:
    print("Failed to load data. Check the file path.")
    exit(1)

# Simple train/test split: first 80% train, last 20% test
split = len(data) * 80 // 100
train = data[:split]
test = data[split:]

print(f"Loaded {len(train)} train, {len(test)} test\n")

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

# --- HNSW ---
print("\n--- HNSW ---")

for M, ef_s in [(4, 10), (8, 20), (16, 50)]:
    t0 = time.perf_counter()
    hnsw = HNSWIndex(M=M, ef_construction=200, ef_search=ef_s)
    hnsw.build(train)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = hnsw.evaluate(test, k=5)
    query_time = time.perf_counter() - t0

    print(f"  M={M:2d} ef_search={ef_s:2d}  accuracy={acc * 100:.1f}%  "
          f"build={build_time * 1000:.1f}ms  query={query_time * 1000:.1f}ms")

# =============================================================
# MNIST Benchmark (5000 train, 1000 test, 784 dimensions)
# =============================================================
print("\n" + "=" * 60)
print("MNIST Benchmark (5000 train, 1000 test, 784 dims)")
print("=" * 60)

mnist_train = load_mnist("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")[:5000]
mnist_test = load_mnist("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")[:1000]
print(f"Loaded {len(mnist_train)} train, {len(mnist_test)} test\n")

results = []

# --- KNN brute-force ---
print("KNN (brute-force, k=5)...")
t0 = time.perf_counter()
knn_acc = evaluate(mnist_test, mnist_train, k=5)
knn_time = time.perf_counter() - t0
print(f"  accuracy={knn_acc * 100:.1f}%  time={knn_time * 1000:.0f}ms")
results.append(("KNN (brute)", knn_acc, 0, knn_time))

# --- Annoy ---
print("\nAnnoy (n_trees=10, k=5)...")
t0 = time.perf_counter()
annoy_idx = AnnoyIndex(n_trees=10, max_leaf_size=10)
annoy_idx.build(mnist_train)
annoy_build = time.perf_counter() - t0

t0 = time.perf_counter()
annoy_acc = annoy_idx.evaluate(mnist_test, k=5)
annoy_query = time.perf_counter() - t0
print(f"  accuracy={annoy_acc * 100:.1f}%  build={annoy_build * 1000:.0f}ms  query={annoy_query * 1000:.0f}ms")
results.append(("Annoy", annoy_acc, annoy_build, annoy_query))

# --- HNSW ---
print("\nHNSW (M=16, ef_search=50, k=5)...")
t0 = time.perf_counter()
hnsw_idx = HNSWIndex(M=16, ef_construction=200, ef_search=50)
hnsw_idx.build(mnist_train)
hnsw_build = time.perf_counter() - t0

t0 = time.perf_counter()
hnsw_acc = hnsw_idx.evaluate(mnist_test, k=5)
hnsw_query = time.perf_counter() - t0
print(f"  accuracy={hnsw_acc * 100:.1f}%  build={hnsw_build * 1000:.0f}ms  query={hnsw_query * 1000:.0f}ms")
results.append(("HNSW", hnsw_acc, hnsw_build, hnsw_query))

# --- Summary table ---
print(f"\n{'Algorithm':<16} {'Accuracy':>8}  {'Build':>9}  {'Query':>9}  {'Total':>9}")
print("-" * 60)
for name, acc, build, query in results:
    total = build + query
    build_str = f"{build * 1000:.0f}ms" if build > 0 else "—"
    print(f"{name:<16} {acc * 100:>7.1f}%  {build_str:>9}  {query * 1000:>8.0f}ms  {total * 1000:>8.0f}ms")
