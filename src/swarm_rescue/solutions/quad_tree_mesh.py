from dataclasses import dataclass
from typing import Iterator, List, Optional, Self

import networkx as nx
import numpy as np

EPS = 10  # QuadTreeMesh explained subdivide


# class Point(np.ndarray):
#     """Represent a point in the 2D space where x and y start at the bottom left corner"""

#     def __new__(cls, x: int, y: int, dtype=np.int_):
#         # python ints are arbitrary precision ints
#         # so when transforming into a np.ndarray,
#         # overflow may happen
#         obj = np.asarray([x, y], dtype=dtype).view(cls)
#         return obj

#     # forces operations on Points to return a Point or a scalar
#     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
#         # transform np.ndarray into Point and keep scalars
#         inputs_arr = [x.view(np.ndarray) if isinstance(x, Point) else x for x in inputs]

#         result = getattr(ufunc, method)(*inputs_arr, **kwargs)

#         if isinstance(result, np.ndarray):
#             return result.view(Point)
#         else:
#             # keep scalar
#             return result

#     def __str__(self):
#         return f"Point({self[0]}, {self[1]})"

#     def __repr__(self):
#         return f"Point({self[0]}, {self[1]})"

#     @property
#     def x(self) -> int:
#         return self[0]

#     @property
#     def y(self) -> int:
#         return self[1]

#     def l_inf_distance_to(self, other: Self) -> int:
#         return max(abs(self - other))


class QuadTreeMesh:
    def __init__(
        self,
        center: np.ndarray,
        width: int,
        graph: nx.Graph,
        # root: Optional[Self] = None,
    ):
        self.center = center
        self.width = width
        self.graph = graph
        # self.root = root if root else self
        self.points: List[np.ndarray] = []

        # Store children as a list [SW, SE, NW, NE]
        self.children: List["QuadTreeMesh"] = []

        # graph.add_node(self)
        # neighbors = self.root.find_neighbors(self)
        # for node in neighbors:
        #     inter = self.intersection(node)

    def insert(self, point: np.ndarray):
        """
        Insert a point into the quadtree.

        Be certain to only give points that are actually inside this
        square otherwise it could lead to unexepected behaviour.
        """

        node = self.find(point)

        # we are using the L inf distance
        if abs(point - node.center).max() >= node.width // 2 - EPS:
            print("OK")
            node.points.append(point)
        else:
            print("NOT OK")
            node.subdivide()
            node.insert(point)

    def find(self, point: np.ndarray) -> "QuadTreeMesh":
        """Get the node where point resides"""

        if self.subdivided:
            index = (point[1] >= self.center[1]) * 2 + (point[0] >= self.center[0])
            return self.children[index].find(point)
        else:
            return self

    def subdivide(self):
        """Subdivide this quadtree into 4 children"""

        centers = [
            np.array(
                [
                    self.center[0] - self.width // 4,
                    self.center[1] - self.width // 4,
                ]
            ),
            np.array(
                [
                    self.center[0] + self.width // 4,
                    self.center[1] - self.width // 4,
                ]
            ),
            np.array(
                [
                    self.center[0] - self.width // 4,
                    self.center[1] + self.width // 4,
                ]
            ),
            np.array(
                [
                    self.center[0] + self.width // 4,
                    self.center[1] + self.width // 4,
                ]
            ),
        ]

        self.children = [
            QuadTreeMesh(center, self.width // 2, self.graph) for center in centers
        ]

        # Redistribute existing points to children
        for point in self.points:
            self.insert(point)

        # self.graph.remove_node(self)
        self.points.clear()

    def find_neighbors(self, target: "QuadTreeMesh") -> List["QuadTreeMesh"]:
        """find all the direct neighbors where edges touch in more than a point of the target inside self"""

        # an intersection being a point does not count as an edge touch
        # it needs non finite intersection
        def edges_touch(A: "QuadTreeMesh", B: "QuadTreeMesh"):
            Amin = A.center - A.width // 2
            Amax = A.center + A.width // 2
            Bmin = B.center - B.width // 2
            Bmax = B.center + B.width // 2

            overlap = np.minimum(Amax, Bmax) - np.maximum(Amin, Bmin)

            return (
                (overlap[0] > 0 and overlap[1] >= 0)  # X edges touch
                or (overlap[0] >= 0 and overlap[1] > 0)  # Y edges touch
            )

        res = []
        stack: List["QuadTreeMesh"] = [self]

        while len(stack) != 0:
            node = stack.pop()
            if edges_touch(node, target):
                if node.subdivided:
                    stack.extend(node.children)
                elif node is not target:
                    res.append(node)

        return res

    def get_leaf_nodes(self) -> List[Self]:
        """Get all leaf nodes (squares without children)"""
        if not self.subdivided:
            return [self]

        leaves = []

        for child in self.children:
            leaves.extend(child.get_leaf_nodes())

        return leaves

    # def intersection(self, other: Self):
    #     Amin = self.center - self.width // 2
    #     Amax = self.center + self.width // 2
    #     Bmin = other.center - other.width // 2
    #     Bmax = other.center + other.width // 2

    #     overlap_box = (np.maximum(Amin, Bmin), np.minimum(Amax, Bmax))

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
