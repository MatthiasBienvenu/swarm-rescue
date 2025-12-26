from typing import List, Optional, Self

import networkx as nx
import numpy as np

EPS = 10  # has to be a multiple of 2


class QuadTreeMesh:
    def __init__(
        self,
        center: np.ndarray,
        width: int,
        graph: nx.Graph,
        root: Optional[Self],
    ):
        self.center = center
        self.width = width
        self.graph = graph
        self.root = root if root else self
        self.points: List[np.ndarray] = []

        # store children as a list [SW, SE, NW, NE]
        self.children: List["QuadTreeMesh"] = []

        print(f"added the node {self}")
        graph.add_node(self)

    def insert(self, point: np.ndarray) -> "QuadTreeMesh":
        """
        Insert a point into the quadtree.

        Be certain to only give points that are actually inside this
        square otherwise it could lead to unexepected behaviour.

        returns the node the point got inserted in
        """

        node = self.find(point)

        # we are using the L inf distance
        if abs(point - node.center).max() >= node.width // 2 - EPS:
            print("OK")
            node.points.append(point)
            return node
        else:
            print("NOT OK")
            node.subdivide()
            return node.insert(point)

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
            QuadTreeMesh(center, self.width // 2, self.graph, self.root)
            for center in centers
        ]

        for point in self.points:
            self.insert(point)

        self.graph.remove_node(self)
        self.points.clear()

        for child in self.children:
            child.add_all_neighbors()

        for child in self.children:
            child.remove_illegal_edges()

    def add_all_neighbors(self):
        """adds all the edges from self to its neighbors"""

        neighbors = self.root.find_neighbors(self)

        for n in neighbors:
            if not self.graph.has_edge(self, n):
                self.graph.add_edge(
                    self,
                    n,
                    weight=np.sqrt(((n.center - self.center) ** 2).sum()),
                )

    def remove_illegal_edges(self):
        """removes all illegal edges that from self to another node"""

        for other in list(self.graph.neighbors(self)):
            box = self.overlap(other)
            assert box is not None

            overlap = box[1] - box[0]
            assert 0 in overlap

            # expand the box by to a width of EPS
            box[:, overlap.tolist().index(0)] += [-EPS, EPS]

            for p in self.points + other.points:
                if all((box[0] <= p) & (p <= box[1])):
                    self.graph.remove_edge(self, other)
                    break

    def find_neighbors(self, target: "QuadTreeMesh") -> List["QuadTreeMesh"]:
        """find all the direct neighbors where edges touch in more than a point of the target inside self"""

        res = []
        stack: List["QuadTreeMesh"] = [self]

        while len(stack) != 0:
            node = stack.pop()
            if node.overlap(target) is not None:
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

    def overlap(self, other: "QuadTreeMesh"):
        Amin = self.center - self.width // 2
        Amax = self.center + self.width // 2
        Bmin = other.center - other.width // 2
        Bmax = other.center + other.width // 2

        box = np.array([np.maximum(Amin, Bmin), np.minimum(Amax, Bmax)])

        return (
            box
            # the overlap is more than just a single point
            if all(box[0] <= box[1]) and not all(box[0] == box[1])
            else None
        )

    def __hash__(self) -> int:
        return hash(tuple(self.center))

    def __str__(self) -> str:
        return f"""Node{{center: ({self.center[0]}, {self.center[1]}), width: {self.width}}}"""

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
