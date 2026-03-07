# HNSW — The Math & Intuition

## The problem with flat search

Brute-force KNN compares the query against every point. Annoy narrows it down with tree splits. HNSW takes a completely different approach: it builds a **graph** where nearby points are connected by edges, then **walks** the graph to find neighbors.

## The small world idea

Imagine you want to find someone in a city. You could check every person (brute force). Or you could ask someone you know, who points you to someone closer to who you're looking for, who points you to someone even closer, and so on. That's a **greedy graph search**.

A "small world" graph has a special property: you can get from any node to any other node in surprisingly few hops. Social networks are like this — "six degrees of separation."

HNSW builds a graph where:
- Every point connects to its nearby neighbors
- You can find any point's nearest neighbors by walking edges greedily

## Why hierarchical?

A flat small world graph has a problem: greedy search can get stuck in local minima, and early hops waste time in dense neighborhoods.

Solution: build **multiple layers**, like a highway system.

```
Layer 2:   o-----------o                     (few nodes, long-range links)
Layer 1:   o-----o-----o-----o               (more nodes, medium links)
Layer 0:   o--o--o--o--o--o--o--o--o--o      (all nodes, short-range links)
```

- **Layer 0** has ALL points, each connected to M nearest neighbors
- **Layer 1** has ~1/M of the points
- **Layer 2** has ~1/M² of the points
- And so on

Higher layers are sparser — each hop covers more distance. Like taking a highway to get close, then local roads to get to the exact address.

## What "small world" means

A small world graph has two properties: (1) nodes that share a neighbor tend to connect to each other (clustering), and (2) any node can reach any other in very few hops (short paths). Social networks are like this — 7 billion people, ~6 hops apart. In HNSW, nearby points form tight clusters, and upper layers add long-range shortcuts. This makes greedy search work — always hopping to the closest neighbor converges instead of getting stuck.

## The key parameters

**M** — max connections per node per layer. Higher M = better recall but more memory and slower build. Think of it as "how many neighbors each person knows."

**ef_construction** — beam width when building the graph. When inserting a new point, how many candidates do we consider before choosing its M neighbors? Higher = better graph quality, slower build.

**ef_search** — beam width when querying. How many candidates do we explore before returning results? Higher = better recall, slower query. Must be >= k.

### Beam width intuition

ef=1 means greedy: only track the single best candidate, always hop to the closest neighbor. Fast but can get stuck if the best path goes through a temporarily-farther node. ef=10 means track 10 candidates — if one path is a dead end, you have 9 others. More work, much better recall. Upper layers use ef=1 (sparse, just get close fast). Layer 0 uses ef=ef_search (dense, need careful exploration).

## How insertion works

When adding a new point p:

1. **Assign a random level L.** Most points get level 0. Some get level 1. Very few get level 2+. The probability decays exponentially: P(level >= L) = (1/M)^L.

2. **Greedy descent from top to L+1.** Starting at the entry point (the highest-level node), greedily walk toward p at each layer, finding the single closest node. This is fast because upper layers are sparse. The result is an entry point close to p.

3. **Connect at layers L down to 0.** At each of these layers:
   - Do a beam search (with width ef_construction) to find candidates near p
   - Pick the M closest candidates as p's neighbors
   - Add bidirectional edges (p->neighbor and neighbor->p)
   - If any neighbor now has more than M connections, trim it (keep only its M closest)

## How query works

To find k nearest neighbors of query q:

1. **Greedy descent from top layer to layer 1.** At each layer, find the single closest node (ef=1). Use it as the entry point for the next layer down. This quickly narrows the search to the right neighborhood.

2. **Beam search at layer 0.** Now do a wider search (ef=ef_search) at the bottom layer where all points live. This explores more carefully to find the actual nearest neighbors.

3. **Return the k closest** from the beam search results.

## Beam search — the core algorithm

Beam search at a single layer maintains two heaps:

- **candidates** (min-heap): nodes to explore, ordered closest-first
- **results** (max-heap): best results so far, ordered farthest-first

```
Initialize both heaps with the entry point(s).
Mark entry points as visited.

While candidates is not empty:
    c = pop closest candidate
    f = peek farthest result

    If dist(c, query) > dist(f, query) and |results| >= ef:
        Stop — no remaining candidate can improve our results.

    For each neighbor n of c in this layer:
        If n already visited: skip
        Mark n as visited
        Compute dist(n, query)

        If |results| < ef or dist(n, query) < dist(f, query):
            Push n onto both candidates and results
            If |results| > ef:
                Pop the farthest result (trim)
```

The stopping condition is the key insight: once the closest unexplored candidate is farther than our worst result, we're done. The graph structure guarantees that continuing would only find worse points.

## Why the heap needs negation in Python

Python's `heapq` is a min-heap only. To simulate a max-heap (for results, where we want to quickly find and pop the farthest), store negated distances:

```
min-heap candidates: (distance, node_id)      -> pop gives closest
max-heap results:    (-distance, node_id)      -> pop gives farthest
```

When you push `(-dist, node_id)` and pop, you get the entry with the largest original distance.

## Why HNSW works so well

1. **Logarithmic descent.** The number of layers is O(log n). Each layer halves the search space. So getting to the right neighborhood takes O(log n) steps.

2. **The graph is navigable.** Within each layer, the small-world property means greedy search converges quickly — typically O(log n) hops.

3. **No wasted work.** Unlike Annoy where you might land in the wrong leaf, HNSW's beam search explores outward from the best known point. The stopping condition ensures you don't explore more than necessary.

4. **Incremental.** Points can be inserted one at a time. No need to rebuild the whole structure (unlike Annoy's trees).

## Comparison to Annoy

| | Annoy | HNSW |
|---|---|---|
| Structure | Forest of binary trees | Multi-layer graph |
| Build | Split data top-down | Insert points one-by-one |
| Query | Traverse trees -> union leaves -> rank | Walk graph -> beam search -> rank |
| Tuning knobs | n_trees, max_leaf_size | M, ef_construction, ef_search |
| Incremental insert | No (must rebuild) | Yes |
| Memory | Lower | Higher (stores graph edges) |
| Recall at same speed | Good | Better |
