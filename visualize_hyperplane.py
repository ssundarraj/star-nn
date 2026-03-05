"""Visualize the perpendicular bisector hyperplane between two points.

Uses the same numbers from the explanation:
  a = [0, 0], b = [4, 0]
  normal = [4, 0], midpoint = [2, 0], offset = 8

Run: python3 visualize_hyperplane.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# --- Points a and b ---
a = [0, 0]
b = [4, 0]
ax.plot(*a, 'ro', markersize=12, zorder=5)
ax.plot(*b, 'bo', markersize=12, zorder=5)
ax.annotate('a = [0, 0]', xy=a, xytext=(-0.5, -0.8), fontsize=11, color='red')
ax.annotate('b = [4, 0]', xy=b, xytext=(3.5, -0.8), fontsize=11, color='blue')

# --- Midpoint ---
m = [2, 0]
ax.plot(*m, 'k^', markersize=10, zorder=5)
ax.annotate('midpoint [2, 0]', xy=m, xytext=(2.1, 0.3), fontsize=9)

# --- Normal vector (drawn from midpoint) ---
ax.annotate('', xy=(3.2, 0), xytext=(2, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.annotate('normal n = [4, 0]\n(direction a→b)', xy=(3.2, 0),
            xytext=(3.3, 0.5), fontsize=9, color='green')

# --- Hyperplane (vertical line at x=2) ---
ax.axvline(x=2, color='black', linestyle='--', linewidth=2, label='hyperplane: n·x = 8')

# --- Shade left and right sides ---
ax.axvspan(-1.5, 2, alpha=0.08, color='red', label='LEFT (n·x - d ≤ 0)')
ax.axvspan(2, 5.5, alpha=0.08, color='blue', label='RIGHT (n·x - d > 0)')

# --- Test points ---
test_points = [
    ([1, 5],  'n·x - d = -4 → LEFT',  'red'),
    ([2, 3],  'n·x - d = 0 → ON LINE', 'black'),
    ([3, 2],  'n·x - d = 4 → RIGHT',   'blue'),
]
for pt, label, color in test_points:
    ax.plot(*pt, 's', color=color, markersize=8, zorder=5)
    ax.annotate(f'{pt}  {label}', xy=pt,
                xytext=(pt[0] + 0.15, pt[1] + 0.15), fontsize=8, color=color)

# --- Formatting ---
ax.set_xlim(-1.5, 5.5)
ax.set_ylim(-2, 6)
ax.set_aspect('equal')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_title('Perpendicular Bisector Hyperplane\nn·x = d  where n = b - a, d = n·midpoint')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperplane.png', dpi=150)
print("Saved to hyperplane.png")
plt.show()
