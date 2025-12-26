import random as rd

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle

from .quad_tree_mesh import EPS, QuadTreeMesh


def main():
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

    root = QuadTreeMesh(np.array([0, 0]), 256, nx.Graph(), None)

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
        root.insert(point).remove_illegal_edges()

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
            c=[
                hsv_to_rgb(((hash(root.find(p)) / 256) % 1, 1, 1))
                for p in points[: i + 1]
            ],
        )

        # draw leaf node boundaries
        leaves = root.get_leaf_nodes()
        for leaf in leaves:
            draw_leaf_with_ok_zone(ax, leaf)

        to_remove = draw_mesh_graph(ax, root.graph)

        plt.draw()
        plt.pause(0.05)

        for p in to_remove:
            p.remove()

    draw_mesh_graph(ax, root.graph)

    leaves = root.get_leaf_nodes()
    for leaf in leaves:
        to_remove = draw_leaf_and_neighbors(ax, root, leaf)

        plt.draw()
        plt.pause(1)

        for p in to_remove:
            p.remove()

    plt.ioff()
    plt.show()


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

        box = target_leaf.overlap(n)
        assert box is not None

        overlap = box[1] - box[0]
        assert 0 in overlap

        # expand the box by to a width of 2 EPS
        box[:, overlap.tolist().index(0)] += [-EPS, EPS]

        if box is not None:
            to_remove.append(
                Rectangle(
                    (box[0, 0], box[0, 1]),
                    box[1, 0] - box[0, 0],
                    box[1, 1] - box[0, 1],
                    fill=True,
                    facecolor="green",
                    edgecolor="green",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.25,
                    zorder=5,
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


def draw_mesh_graph(ax: Axes, graph: nx.Graph):
    pos = {node: node.center for node in graph.nodes}

    to_remove = []
    to_remove.append(
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            ax=ax,
            edge_color="purple",
            width=1.0,
            alpha=0.6,
        )
    )

    to_remove.append(
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            ax=ax,
            node_size=20,
            node_color="purple",
            alpha=0.8,
        )
    )

    return to_remove


if __name__ == "__main__":
    main()
