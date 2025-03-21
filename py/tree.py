import numpy as np
from collections import deque


class TreeNode:

    # Lower and Upper bound
    def __init__(self, L, U):
        self.L = L
        self.U = U
        self.M = (L + U) / 2
        self.left = None
        self.right = None
        self.weights = None


class Tree:

    def __init__(self, root: TreeNode):
        self.root = root

    def print_tree(self) -> None:

        q = deque([self.root])

        level = 0

        while q:
            nodes = []
            for _ in range(len(q)):
                node = q.popleft()
                if node:
                    nodes.append([node.L, node.U, node.weights])
                    q.append(node.left)
                    q.append(node.right)
            if not nodes:
                break
            print(f"Nodes in level: {level}, {nodes}")
            level += 1

        return


def construct_tree(
    T: tuple[int, int],
    k: int,
) -> Tree:
    """Function to build the binary tree code for uniform distribution of unit interval

    Args:
        T: Interval as a tuple of ints consisting of L, U where L and U are the lower and upper bound
        k: The number of levels in the tree
    """

    L, U = T  # extract the lower and upper bounds

    root = TreeNode(L, U)
    tree = Tree(root)

    # building the tree in level order
    q = deque([root])

    while k > 0:
        for _ in range(len(q)):
            node = q.popleft()
            mid = (node.L + node.U) / 2

            node.left = TreeNode(node.L, mid)
            node.right = TreeNode(mid, node.U)

            q.append(node.left)
            q.append(node.right)

        k -= 1

    return tree


def compute_weights(
    node: TreeNode,
    sources: np.array(float),
    charges: np.array(float),
    p: int,
):

    # Base case for NULL Node
    if node is None:
        return

    sources_in_cell = sources[(sources >= node.L) & (sources <= node.U)]
    charges_in_cell = charges[(sources >= node.L) & (sources <= node.U)]

    # Initializing weights for the node
    node.weights = np.zeros(p + 1)

    # Compute weights through multipole expansion

    for m in range(p + 1):
        node.weights[m] = np.sum(charges_in_cell * (sources_in_cell - node.M) ** m)

    # Recurse on left and right subtree to compute weights
    compute_weights(node.left, sources, charges, p)
    compute_weights(node.right, sources, charges, p)

    return


if __name__ == "__main__":

    # Create Tree
    T = (0, 1)
    k = 3
    tree = construct_tree(T, k)

    # Initiate sources and charges
    sources = np.array([0.1, 0.2, 0.6, 0.9])
    charges = np.array([1.0, -1.0, 1.0, -1.0])

    # order of expansion
    p = 2

    compute_weights(tree.root, sources, charges, p)
    tree.print_tree()
