import random as rd

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle

from .quad_tree_mesh import EPS, QuadTreeMesh


def draw_leaf_with_ok_zone(ax, leaf: QuadTreeMesh):
    # draw the outer boundary (blue)
    rect = Rectangle(
        (leaf.center[0] - leaf.width / 2, leaf.center[1] - leaf.width / 2),
        leaf.width,
        leaf.width,
        fill=False,
        edgecolor="blue",
        linewidth=1,
        alpha=0.5,
    )
    ax.add_patch(rect)

    # draw the "ok zone" (green) - where points don't trigger subdivision
    # this is the area where l_inf_distance >= width/2 - EPS
    ok_zone_size = leaf.width // 2 - EPS
    ok_zone_rect = Rectangle(
        (leaf.center[0] - ok_zone_size, leaf.center[1] - ok_zone_size),
        2 * ok_zone_size,
        2 * ok_zone_size,
        fill=False,
        edgecolor=("green" if ok_zone_size >= 0 else "red"),
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(ok_zone_rect)


# helper to draw a leaf and its neighbors
def draw_leaf_and_neighbors(
    ax: Axes,
    mesh: QuadTreeMesh,
    target_leaf: QuadTreeMesh,
):
    to_remove = []
    neighbors = mesh.find_neighbors(target_leaf)

    for n in neighbors:
        to_remove.append(
            Rectangle(
                (n.center[0] - n.width / 2, n.center[1] - n.width / 2),
                n.width,
                n.width,
                fill=False,
                hatch="/",
                edgecolor="orange",
                linewidth=2,
                linestyle="--",
                alpha=0.9,
            )
        )

    # draw the target on top
    to_remove.append(
        Rectangle(
            (
                target_leaf.center[0] - target_leaf.width / 2,
                target_leaf.center[1] - target_leaf.width / 2,
            ),
            target_leaf.width,
            target_leaf.width,
            fill=False,
            hatch="/",
            edgecolor="red",
            linewidth=2,
            linestyle="-",
            alpha=1.0,
        )
    )

    for p in to_remove:
        ax.add_patch(p)

    to_remove.append(
        ax.text(
            target_leaf.center[0],
            target_leaf.center[1],
            "TARGET",
            color="red",
            ha="center",
            va="center",
            fontsize=8,
        )
    )

    return to_remove


print(np.array([1, 2]) + 3)
print(type(np.array([1, 2]) + 3))

plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("QuadTree Mesh Visualization")

mesh = QuadTreeMesh(np.array([0, 0]), 256, nx.Graph())

with open("test.csv", "r") as f:
    lines = f.read().strip().split("\n")

    points = []
    # skip header
    for line in lines[1:]:
        if line.strip():
            x_str, y_str = line.split(",")
            points.append(np.array([int(x_str.strip()), int(y_str.strip())]))

    points = np.array(points)

for i, point in enumerate(points):
    mesh.insert(point)

    # ax.plot(point[0], point[1], "ro", markersize=4)
    ax.set_title(f"QuadTree Mesh Visualization - Point {i + 1}/{len(points)}")

    # Draw quadtree boundaries
    ax.clear()
    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"QuadTree Mesh Visualization - Point {i + 1}/{len(points)}")

    # plot all points up to current
    ax.scatter(
        points[: i + 1, 0],
        points[: i + 1, 1],
        s=4,
        c=[hsv_to_rgb(((hash(mesh.find(p)) / 256) % 1, 1, 1)) for p in points[: i + 1]],
    )

    # draw leaf node boundaries
    leaves = mesh.get_leaf_nodes()
    for leaf in leaves:
        draw_leaf_with_ok_zone(ax, leaf)

    plt.draw()
    plt.pause(0.05)


leaves = mesh.get_leaf_nodes()
for leaf in leaves:
    to_remove = draw_leaf_and_neighbors(ax, mesh, leaf)

    plt.draw()
    plt.pause(1)

    for p in to_remove:
        p.remove()

plt.ioff()
plt.show()
