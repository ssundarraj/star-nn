# Annoy — The Math

## Vectors

A point in space is just a list of numbers. In 2D: `[3, 4]` means "3 units right, 4 units up." In 4D (like Iris features): `[5.1, 3.5, 1.4, 0.2]` — same idea, just more dimensions.

## Dot product

The dot product of two vectors multiplies corresponding elements and sums them:

```
[1, 2] · [3, 4] = 1×3 + 2×4 = 11
```

One key property: the dot product tells you how much two vectors point in the same direction. If the result is 0, they're perpendicular. If positive, they point roughly the same way. If negative, roughly opposite.

## Normal vector

A "normal" to a surface is a vector that sticks straight out from it, perpendicular. Think of a table — the normal points straight up.

For a line/plane/hyperplane, the normal tells you which direction the surface is "facing."

## Hyperplane construction

A hyperplane is a flat surface that divides space in two. In 2D it's a line, in 3D it's a plane, in N-D it's an (N-1)-dimensional surface.

Given two points **a** and **b** in R^n, the perpendicular bisector is the set of all points equidistant from both.

Say you have: `a = [0, 0]` and `b = [4, 0]`.

**Step 1: Normal = b - a**
```
n = [4-0, 0-0] = [4, 0]
```
This points from a toward b (pointing right). The hyperplane will be perpendicular to this — so in this case, the hyperplane is a vertical line.

**Step 2: Midpoint = (a + b) / 2**
```
m = [(0+4)/2, (0+0)/2] = [2, 0]
```
The hyperplane passes through the midpoint, halfway between a and b.

**Step 3: Offset = n · m**

The normal vector alone only gives you a *direction* — it says "the plane is perpendicular to this." But there are infinitely many planes perpendicular to the same direction. Imagine sliding a wall left and right along a hallway — they're all perpendicular to the hallway, but at different positions.

The offset tells you *where* along that direction the plane sits. We want it at the midpoint, so we compute:

```
d = n · m = [4, 0] · [2, 0] = 4×2 + 0×0 = 8
```

Think of `n · x` as asking "how far along the normal direction is point x?" The offset `d` says "the plane is at position 8 along that direction."

- For the midpoint: `n · [2,0] = 8`, which equals `d`. It's on the plane. ✓
- For point a: `n · [0,0] = 0`, which is less than `d = 8`. It's behind the plane.
- For point b: `n · [4,0] = 16`, which is greater than `d = 8`. It's in front of the plane.

So `n · x - d` is like a signed distance: negative means "on a's side," positive means "on b's side," zero means "exactly on the plane."

**The hyperplane equation is:** `n · x = d` — the set of all points where `n · x` equals exactly `d`.

## Side test

For any point x, compute `n · x - d`:

| Point | Calculation | Result | Side |
|-------|------------|--------|------|
| `[0, 0]` (a) | `[4,0]·[0,0] - 8 = 0 - 8` | `-8` | LEFT (a's side) |
| `[4, 0]` (b) | `[4,0]·[4,0] - 8 = 16 - 8` | `8` | RIGHT (b's side) |
| `[2, 3]` | `[4,0]·[2,3] - 8 = 8 - 8` | `0` | ON the hyperplane |
| `[1, 5]` | `[4,0]·[1,5] - 8 = 4 - 8` | `-4` | LEFT |

Rule: `n · x - d ≤ 0` → left, `> 0` → right.

## Why this works for Annoy

Each random pair of points creates a different hyperplane, cutting the data differently. Points that are close together tend to end up on the same side of most splits, so they end up in the same leaf. Points far apart get separated early. That's the core insight of the algorithm.

## Visualization

Run `uv run --with matplotlib visualize_hyperplane.py` to see a plot of the example above.
