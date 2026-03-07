"""Visualize the HNSW algorithm concepts.

Three plots:
  1. The multi-layer structure — how points appear across layers
  2. Greedy descent — how query navigates from top to bottom
  3. Beam search at layer 0 — how candidates expand outward

Run: uv run --with matplotlib visualize_hnsw.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random

random.seed(42)
np.random.seed(42)


def plot_layer_structure():
    """Show how HNSW layers contain fewer and fewer points."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate 20 points in 2D
    points = np.random.rand(20, 2) * 10

    # Assign levels: most get 0, few get 1, fewer get 2
    levels = []
    for _ in range(20):
        level = 0
        while random.random() < 0.25 and level < 2:
            level += 1
        levels.append(level)

    # Force at least one point at level 2
    levels[0] = 2

    layer_names = ["Layer 0 (all points)", "Layer 1 (~1/M points)", "Layer 2 (~1/M² points)"]
    colors = ["#4a90d9", "#e67e22", "#e74c3c"]

    for layer in range(3):
        y_offset = layer * 4

        # Draw layer band
        ax.axhspan(y_offset - 0.5, y_offset + 3, alpha=0.05, color=colors[layer])
        ax.text(-1.5, y_offset + 1.2, layer_names[layer], fontsize=9,
                fontweight='bold', color=colors[layer], va='center')

        # Points in this layer
        layer_points = [(i, points[i]) for i in range(20) if levels[i] >= layer]

        # Draw edges between nearby points in this layer
        for i, (idx_a, pa) in enumerate(layer_points):
            for idx_b, pb in layer_points[i+1:]:
                dist = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                max_dist = 3.0 + layer * 2  # higher layers connect farther
                if dist < max_dist:
                    ax.plot([pa[0], pb[0]], [pa[1] + y_offset, pb[1] + y_offset],
                            '-', color=colors[layer], alpha=0.2, lw=1)

        # Draw points
        for idx, p in layer_points:
            ax.plot(p[0], p[1] + y_offset, 'o', color=colors[layer],
                    markersize=8, zorder=5)
            ax.annotate(str(idx), xy=(p[0], p[1] + y_offset),
                        fontsize=6, ha='center', va='center', color='white',
                        fontweight='bold')

    ax.set_xlim(-3, 11)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.set_title('HNSW Layer Structure\nHigher layers have fewer nodes with longer-range connections')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('hnsw_layers.png', dpi=150)
    print("Saved hnsw_layers.png")


def plot_greedy_descent():
    """Show how a query descends through layers."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Fixed points for clarity
    all_points = {
        0: (1, 1), 1: (3, 2), 2: (5, 1.5), 3: (7, 2),
        4: (2, 0.5), 5: (4, 0.8), 6: (6, 0.5), 7: (8, 1),
        8: (8.5, 1.8), 9: (9, 1),
    }
    # Which points exist at each layer
    layer_membership = {
        0: list(range(10)),  # all
        1: [0, 2, 3, 8],
        2: [0, 8],
    }

    query = (8.8, 1.5)
    colors = ["#4a90d9", "#e67e22", "#e74c3c"]

    # Path the query takes: entry at node 0 (layer 2), greedy to 8,
    # then at layer 1: 8->3->8 (already closest), then layer 0: 8->9
    descent_path = [
        (2, 0, 8, "hop to closest"),
        (1, 8, 8, "already closest"),
        (0, 8, 9, "find true nearest"),
    ]

    for layer in range(3):
        y_off = layer * 4

        ax.axhspan(y_off - 0.5, y_off + 3, alpha=0.05, color=colors[layer])
        ax.text(-0.8, y_off + 1.5, f"Layer {layer}", fontsize=9,
                fontweight='bold', color=colors[layer])

        members = layer_membership[layer]
        for idx in members:
            p = all_points[idx]
            ax.plot(p[0], p[1] + y_off, 'o', color=colors[layer],
                    markersize=10, zorder=5)
            ax.annotate(str(idx), xy=(p[0], p[1] + y_off),
                        fontsize=7, ha='center', va='center',
                        color='white', fontweight='bold')

            # Draw some edges
            for other in members:
                if other > idx:
                    op = all_points[other]
                    dist = abs(p[0]-op[0]) + abs(p[1]-op[1])
                    if dist < 4 + layer * 3:
                        ax.plot([p[0], op[0]], [p[1]+y_off, op[1]+y_off],
                                '-', color=colors[layer], alpha=0.15, lw=1)

    # Draw query point at each layer
    for layer in range(3):
        y_off = layer * 4
        ax.plot(query[0], query[1] + y_off, '*', color='green',
                markersize=15, zorder=6)

    # Draw descent arrows
    arrow_style = dict(arrowstyle='->', color='green', lw=2.5)

    # Layer 2: 0 -> 8
    p0 = all_points[0]
    p8 = all_points[8]
    ax.annotate('', xy=(p8[0], p8[1]+8), xytext=(p0[0], p0[1]+8),
                arrowprops=arrow_style)
    ax.text(4.5, 10.5, '1. greedy hop', fontsize=8, color='green', style='italic')

    # Layer 2 -> Layer 1 (vertical descent)
    ax.annotate('', xy=(p8[0]-0.2, p8[1]+4+0.3), xytext=(p8[0]-0.2, p8[1]+8-0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    ax.text(8.8, 6, 'descend', fontsize=7, color='gray', style='italic')

    # Layer 1 -> Layer 0 (vertical descent)
    ax.annotate('', xy=(p8[0]-0.2, p8[1]+0.3), xytext=(p8[0]-0.2, p8[1]+4-0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    ax.text(8.8, 2, 'descend', fontsize=7, color='gray', style='italic')

    # Layer 0: 8 -> 9
    p9 = all_points[9]
    ax.annotate('', xy=(p9[0], p9[1]), xytext=(p8[0], p8[1]),
                arrowprops=arrow_style)
    ax.text(8.3, 0.3, '2. beam search\n    finds #9', fontsize=8,
            color='green', style='italic')

    ax.plot([], [], '*', color='green', markersize=12, label='query')
    ax.legend(loc='lower left')
    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.set_title('HNSW Query: Greedy Descent + Beam Search\nStart at top layer, descend to layer 0, search locally')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('hnsw_descent.png', dpi=150)
    print("Saved hnsw_descent.png")


def plot_beam_search():
    """Show how beam search expands outward at layer 0."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Points arranged in a cluster
    points = {
        0: (2, 5), 1: (3, 6), 2: (4, 5.5), 3: (5, 6),
        4: (3, 4), 5: (4, 4), 6: (5, 4.5), 7: (6, 5),
        8: (7, 5), 9: (4.5, 3), 10: (6, 3.5), 11: (7, 4),
    }

    # Graph edges (adjacency)
    edges = {
        0: [1, 4], 1: [0, 2, 4], 2: [1, 3, 5, 6], 3: [2, 7],
        4: [0, 1, 5, 9], 5: [2, 4, 6, 9], 6: [2, 5, 7, 10],
        7: [3, 6, 8, 11], 8: [7, 11], 9: [4, 5, 10],
        10: [6, 9, 11], 11: [7, 8, 10],
    }

    query = (6.5, 4.8)

    # Draw all edges
    for node, neighbors in edges.items():
        for n in neighbors:
            if n > node:
                p1, p2 = points[node], points[n]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        '-', color='#cccccc', lw=1, zorder=1)

    # Simulate beam search steps from entry point 0
    steps = [
        {"visited": {0}, "candidates": {0}, "label": "Step 1: start at entry"},
        {"visited": {0, 1, 4}, "candidates": {1, 4}, "label": "Step 2: explore neighbors of 0"},
        {"visited": {0, 1, 2, 4, 5, 9}, "candidates": {2, 5}, "label": "Step 3: explore neighbors of 1, 4"},
        {"visited": {0, 1, 2, 3, 4, 5, 6, 9}, "candidates": {6}, "label": "Step 4: getting closer..."},
        {"visited": {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}, "candidates": {7}, "label": "Step 5: found the neighborhood"},
        {"visited": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, "candidates": {8, 11}, "label": "Step 6: converged"},
    ]

    # Show step 5 (most interesting)
    step = steps[4]

    # Color nodes by status
    for idx, p in points.items():
        if idx in step["candidates"]:
            color = '#e74c3c'
            size = 14
        elif idx in step["visited"]:
            color = '#4a90d9'
            size = 11
        else:
            color = '#cccccc'
            size = 9
        ax.plot(p[0], p[1], 'o', color=color, markersize=size, zorder=5)
        ax.annotate(str(idx), xy=p, fontsize=7, ha='center', va='center',
                    color='white', fontweight='bold', zorder=6)

    # Draw edges that were traversed
    for idx in step["visited"]:
        for n in edges[idx]:
            if n in step["visited"]:
                p1, p2 = points[idx], points[n]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        '-', color='#4a90d9', lw=1.5, alpha=0.4, zorder=2)

    # Query
    ax.plot(*query, '*', color='green', markersize=20, zorder=7)
    ax.annotate('query', xy=query, xytext=(query[0]+0.3, query[1]+0.3),
                fontsize=10, color='green', fontweight='bold')

    # Highlight the closest found so far
    ax.annotate('closest candidate\n(current best)', xy=points[7],
                xytext=(7.5, 6), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                color='#e74c3c')

    # Legend
    ax.plot([], [], 'o', color='#4a90d9', markersize=8, label='visited')
    ax.plot([], [], 'o', color='#e74c3c', markersize=8, label='current candidates')
    ax.plot([], [], 'o', color='#cccccc', markersize=8, label='not yet reached')
    ax.plot([], [], '*', color='green', markersize=12, label='query')
    ax.legend(loc='upper left', fontsize=9)

    ax.set_xlim(1, 8.5)
    ax.set_ylim(2.5, 7)
    ax.set_aspect('equal')
    ax.set_title(f'Beam Search at Layer 0\n{step["label"]}')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('hnsw_beam_search.png', dpi=150)
    print("Saved hnsw_beam_search.png")


plot_layer_structure()
plot_greedy_descent()
plot_beam_search()
print("\nDone! Generated 3 images: hnsw_layers.png, hnsw_descent.png, hnsw_beam_search.png")
