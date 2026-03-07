# Learning KNN & ANN algorithms with Claude Code

A log of the key prompts and responses from this session.

## 1. Project setup

**Prompt:** I want to learn how to implement knn in cpp. Bootstrap a repo for me. I don't want you to implement anything since I want to learn by implementation.

**Response:** Created a full C++ project with CMakeLists.txt, stub files with TODO markers, tests for each step, and the Iris dataset. Each function had hints tailored for a beginner.

**Prompt:** I changed my mind. I want to do this in python. Remove everything except the data and start again. I'm an experienced engineer. Tailor the content to me correctly.

**Response:** Deleted all C++ files, recreated stubs in Python with just signatures and step numbers — no syntax tutorials. Gave a concise table of what to implement in what order.

## 2. KNN implementation (user-driven)

**Prompt:** Move knn.py to knn_demo.py and create a blank file for me with just the signature for `evaluate` and `predict`.

**Response:** Renamed the file, created a minimal knn.py with just the two function signatures.

The user then implemented functions themselves, asking targeted Python questions along the way:
- Can I destructure in a for loop? → Yes: `for features, label in data`
- Is there insertion sort? → No, use `sorted()` or `.sort()`
- How do I annotate a type? → `total: float = 0.0`
- `defaultdict(int)` for zero-default counters
- `reduce` with a mutating accumulator — learned it needs a real function, not a lambda with statements
- `max(dict, key=dict.get)` to get the key with the highest value

**Prompt:** Review my code.

**Response:** Found a performance issue in `find_k_nearest` (sorting on every iteration), suggested a simpler sort-once approach. Noted that `reduce` with mutation is unidiomatic Python — `Counter.most_common()` is cleaner. Added review comments directly in the code prefixed with `[claude review]`.

## 3. Annoy — choosing the algorithm

**Prompt:** I want to now implement some ANN algorithms. List several in a table with their ease of implementation and a summary. Also tell me which ones are actually used today.

**Response:** Table of 7 algorithms (KD-Tree, Ball Tree, LSH, VP-Tree, Annoy, HNSW, R-Tree) with ease ratings, summaries, and production usage. Highlighted that HNSW dominates (powers pgvector, Pinecone, etc.), Annoy is used at Spotify, LSH for deduplication.

**Prompt:** Let's build Annoy.

## 4. Annoy implementation

**Response:** Created `annoy.py` with 8 steps of stubs, `test_annoy.py` with tests, updated `main.py` with benchmarks. Used same pattern: signatures + docstrings, no implementation.

**Prompt:** The hints are too helpful. Give me the math without the code.

**Response:** Rewrote all docstrings to describe the algorithm mathematically instead of giving Python code hints.

## 5. Understanding the math

**Prompt:** Explain the math in `_make_split_hyperplane`. I'm not that familiar with vector math.

**Response:** Built up from basics — what vectors are, dot products, normal vectors — then walked through a concrete example with `a=[0,0]`, `b=[4,0]`. Showed how the normal, midpoint, and offset define the hyperplane, with a table of test points and their side assignments.

**Prompt:** Why is `n · m` the offset? Why does dot product work that way?

**Response:** Explained dot product as projection — "shining a flashlight along the normal and looking at shadows." Showed that `[2,0]` and `[2,3]` project to the same value because movement perpendicular to the normal doesn't matter. The offset is just "where the midpoint's shadow lands."

**Prompt:** Visualize the projection stuff.

**Response:** Created `visualize_projection.py` — matplotlib plot showing points with dashed shadow lines projecting onto the normal direction as a number line.

**Prompt:** Write the explanation to annoy-math.md.

**Response:** Created `annoy-math.md` with the full explanation: vectors, dot product, normal vectors, hyperplane construction with worked example, side test table, and why it works for Annoy.

## 6. HNSW

**Prompt:** Set up the files for HNSW.

**Response:** Created `hnsw.py` with 8 steps (constructor, random level, beam search, select neighbors, insert, build, query, evaluate), `test_hnsw.py`, updated `main.py`. The beam search step (step 3) is the hardest — it's the core algorithm.

**Prompt:** Explain HNSW to me in a md file.

**Response:** Created `hnsw-explained.md` covering: the graph-walking intuition, why hierarchical (highway analogy), parameters (M, ef_construction, ef_search), insertion algorithm, query algorithm, beam search pseudocode, the Python heap negation trick, and a comparison table with Annoy.

**Prompt:** What does "small world" mean? Explain beam width more.

**Response:** Small world = high clustering + short paths (six degrees of separation). Beam width = how many candidates you keep alive. ef=1 is greedy (fast, can get stuck), ef=10 explores 10 paths (slower, better recall). Added both to the md file.

## 7. HNSW implementation (user-driven)

The user implemented HNSW step by step, with Claude reviewing after each function. Key debugging moments:

- **`_search_layer`**: Used the full tuple instead of `c[1]` for layer lookup (KeyError). Forgot to add entry points to the `visited` set. Return value wasn't sorted.
- **`insert`**: `if not self._entry_point` fails when entry_point is 0 — changed to `is None`. Typo `self.max_level` (missing underscore). Connection guard `node_layer <= cur_layer` was flipped — should be `>=`. First node returned early before registering in layers or setting `_max_level`. `cur_layer -= 1` was placed before the connection logic.
- **`query`**: `cur_layer > 1` should be `> 0`. Missing `max(ef_search, k)`. Missing `cur_layer -= 1` caused infinite loop.

Each bug was found through code review, not by Claude writing the code. The user fixed issues themselves after Claude pointed them out.

**Prompt:** Explain the random level math to me in detail.

**Response:** Walked through the probability distribution: P(level >= L) = (1/M)^L. Explained why `floor(-ln(r) / ln(M))` produces this — inverse CDF of a geometric distribution. Helped visualize it in Desmos.

**Prompt:** What are the entry_points in `_search_layer`? Is there only 1 node at the top layer?

**Response:** Not necessarily one node — any node whose assigned level reaches that layer exists there. But the entry_point is a single designated node (the one with the highest level at time of insertion). During query, you start at this one node and greedily descend.

All 20 HNSW tests passed with 80% accuracy on Iris.

## 8. Visualizations

**Response:** Created three visualization scripts throughout:
- `visualize_hyperplane.py` — the perpendicular bisector with shaded sides
- `visualize_projection.py` — dot product as projection onto normal direction
- `visualize_hnsw.py` — layer structure, greedy descent, and beam search expansion

## 9. MNIST benchmark

**Prompt:** Find a better dataset with more data to evaluate annoy, hnsw, knn.

**Response:** Presented options (MNIST, Pen Digits, Letter Recognition, Wine Quality). User chose MNIST.

**Prompt:** Can you use binary format for the MNIST dataset and write a new loader for it?

**Response:** Downloaded the original IDX binary files (gzipped) from Google's MNIST mirror. Wrote `mnist_loader.py` that reads the binary format directly using `struct.unpack` and `gzip`. Returns the same `(features, label)` tuple format as `csv_loader.py`. 60K train, 10K test, 784 features normalized to [0, 1].

**Prompt:** Update main.py to benchmark all three on MNIST.

**Response:** Added a MNIST section to `main.py` with 5K train / 1K test. Single config per algorithm (no parameter sweeps — too slow at 784 dims). Comparison summary table at the end.

## 10. Performance investigation

**Prompt:** Why is KNN so slow? Might it have something to do with the sort on line 60?

**Response:** Yes — `sorted(k_nearest)` on every iteration is O(n k log k). Suggested a max-heap approach using `heapq` (same trick as HNSW's beam search). Push `-dist` for max-heap behavior, `heapreplace` to swap out the farthest candidate. Total: O(n log k) instead of O(n k log k).

**Prompt:** Why is it still taking forever on 5000/1000?

**Response:** The bottleneck is `euclidean_distance` — a pure Python loop over 784 floats, called millions of times. Tried `numpy` (`np.linalg.norm`) but calling `np.asarray()` on every distance call added overhead. The real fix would be converting data to numpy arrays once upfront in the loader. Tested it — much faster, but user reverted since it's a learning project.

Discussed accuracy of benchmark numbers: no warmup, single run, wall-clock time. Fine for seeing relative differences, not for precise measurement.
