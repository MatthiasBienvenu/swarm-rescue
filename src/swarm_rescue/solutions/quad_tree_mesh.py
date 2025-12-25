from dataclasses import dataclass
from typing import Iterator, List, Self

import networkx as nx
import numpy as np

EPS = 10  # QuadTreeMesh explained subdivide


class Point(np.ndarray):
    """Represent a point in the 2D space where x and y start at the bottom left corner"""

    def __new__(cls, x: int, y: int, dtype=np.int_):
        # python ints are arbitrary precision ints
        # so when transforming into a np.ndarray,
        # overflow may happen
        obj = np.asarray([x, y], dtype=dtype).view(cls)
        return obj

    # forces operations on Points to return a Point or a scalar
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # transform np.ndarray into Point and keep scalars
        inputs_arr = [x.view(np.ndarray) if isinstance(x, Point) else x for x in inputs]

        result = getattr(ufunc, method)(*inputs_arr, **kwargs)

        if isinstance(result, np.ndarray):
            return result.view(Point)
        else:
            # keep scalar
            return result

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    def l_inf_distance_to(self, other: Self) -> int:
        return max(abs(self - other))


class QuadTreeMesh:
    def __init__(
        self,
        center: Point,
        width: int,
        graph: nx.Graph,
    ):
        self.center = center
        self.width = width
        self.graph = graph
        self.points: List[Point] = []

        # Store children as a list [SW, SE, NW, NE]
        self.children: List["QuadTreeMesh"] = []

    def find(self, point: Point) -> "QuadTreeMesh":
        """Get the node where point resides"""

        if self.subdivided:
            index = (point.y >= self.center.y) * 2 + (point.x >= self.center.x)
            return self.children[index].find(point)
        else:
            return self

    def find_neighbors(self, target: "QuadTreeMesh") -> List["QuadTreeMesh"]:
        """find all the direct neighbors of the target inside self"""

        def intersect(A: "QuadTreeMesh", B: "QuadTreeMesh"):
            Amin = A.center - A.width // 2
            Amax = A.center + A.width // 2
            Bmin = B.center - B.width // 2
            Bmax = B.center + B.width // 2

            overlap_x = min(Amax.x, Bmax.x) - max(Amin.x, Bmin.x)
            overlap_y = min(Amax.y, Bmax.y) - max(Amin.y, Bmin.y)

            return not (
                Amax.x < Bmin.x or Amin.x > Bmax.x or Amax.y < Bmin.y or Amin.y > Bmax.y
            )

        res = []
        stack: List["QuadTreeMesh"] = [self]

        while len(stack) != 0:
            node = stack.pop()
            if intersect(node, target):
                if node.subdivided:
                    stack.extend(node.children)
                else:
                    res.append(node)

        return res

    def insert(self, point: Point):
        """
        Insert a point into the quadtree.

        Be certain to only give points that are actually inside this
        square otherwise it could lead to unexepected behaviour.
        """

        node = self.find(point)

        if point.l_inf_distance_to(node.center) >= node.width // 2 - EPS:
            print("OK")
            node.points.append(point)
        else:
            print("NOT OK")
            node.subdivide()
            node.insert(point)

    def subdivide(self):
        """Subdivide this quadtree into 4 children"""

        centers = [
            Point(
                self.center.x - self.width // 4,
                self.center.y - self.width // 4,
            ),
            Point(
                self.center.x + self.width // 4,
                self.center.y - self.width // 4,
            ),
            Point(
                self.center.x - self.width // 4,
                self.center.y + self.width // 4,
            ),
            Point(
                self.center.x + self.width // 4,
                self.center.y + self.width // 4,
            ),
        ]

        self.children = [
            QuadTreeMesh(center, self.width // 2, self.graph) for center in centers
        ]

        # Redistribute existing points to children
        for point in self.points:
            self.insert(point)

        self.points.clear()

    def get_leaf_nodes(self) -> List[Self]:
        """Get all leaf nodes (squares without children)"""
        if not self.subdivided:
            return [self]

        leaves = []

        for child in self.children:
            leaves.extend(child.get_leaf_nodes())

        return leaves

    @property
    def subdivided(self) -> bool:
        return len(self.children) == 4

    @property
    def southwest(self) -> "QuadTreeMesh":
        return self.children[0]

    @property
    def southeast(self) -> "QuadTreeMesh":
        return self.children[1]

    @property
    def northwest(self) -> "QuadTreeMesh":
        return self.children[2]

    @property
    def northeast(self) -> "QuadTreeMesh":
        return self.children[3]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.patches import Rectangle

    print(Point(1, 2) + 3)
    print(type(Point(1, 2) + 3))

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("QuadTree Mesh Visualization")

    mesh = QuadTreeMesh(Point(0, 0), 256, nx.Graph())

    with open("test.csv", "r") as f:
        lines = f.read().strip().split("\n")

        points = []
        # skip header
        for line in lines[1:]:
            if line.strip():
                x_str, y_str = line.split(",")
                points.append(Point(int(x_str.strip()), int(y_str.strip())))

    for i, point in enumerate(points):
        mesh.insert(point)

        # ax.plot(point.x, point.y, "ro", markersize=4)
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
            [p.x for p in points[: i + 1]],
            [p.y for p in points[: i + 1]],
            s=4,
            c=[
                hsv_to_rgb(((hash(mesh.find(p)) / 256) % 1, 1, 1))
                for p in points[: i + 1]
            ],
        )

        # draw leaf node boundaries
        leaves = mesh.get_leaf_nodes()
        for leaf in leaves:
            # draw the outer boundary (blue)
            rect = Rectangle(
                (leaf.center.x - leaf.width / 2, leaf.center.y - leaf.width / 2),
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
                (leaf.center.x - ok_zone_size, leaf.center.y - ok_zone_size),
                2 * ok_zone_size,
                2 * ok_zone_size,
                fill=False,
                edgecolor=("green" if ok_zone_size >= 0 else "red"),
                linewidth=1,
                linestyle="--",
                alpha=0.7,
            )
            ax.add_patch(ok_zone_rect)

        plt.draw()
        plt.pause(0.005)

    plt.ioff()
    plt.show()
