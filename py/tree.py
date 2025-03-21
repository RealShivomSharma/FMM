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
    """
    Base Tree Class for Building uniform distribution
    """

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

            node.left = TreeNode(node.L, node.M)
            node.right = TreeNode(node.M, node.U)

            q.append(node.left)
            q.append(node.right)

        k -= 1

    return tree


def compute_weights_recursive(
    node: TreeNode,
    sources: np.array(float),
    charges: np.array(float),
    p: int,
):
    """Multipole weight computation using Depth First Traversal

    Args:
        node: Current TreeNode
        sources: Source vector
        charges: Charge vector
        p: Order of expansion
    """
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
    compute_weights_recursive(node.left, sources, charges, p)
    compute_weights_recursive(node.right, sources, charges, p)

    return


def compute_weights_iterative(
    root: TreeNode,
    sources: np.array(float),
    charges: np.array(float),
    p: int,
):
    """Multipole weight computation using level-order traversal

    Args:
        root: Root TreeNode of the tree to compute
        sources: Source vector
        charges: Charge vector
        p: Order of expansion
    """

    q = deque([root])

    while q:
        node = q.popleft()

        sources_in_cell = sources[(sources >= node.L) & (sources <= node.U)]
        charges_in_cell = charges[(sources >= node.L) & (sources <= node.U)]

        # Initializing weights for the node
        node.weights = np.zeros(p + 1)

        if len(sources_in_cell) > 0:
            for m in range(p + 1):
                node.weights[m] = np.sum(
                    charges_in_cell * (sources_in_cell - node.M) ** m
                )

        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return


def far_field_approximation(tree: Tree, delta: float, targets: np.array, p: int):
    """Function to compute the far field approximation

    Args:
        tree: Tree Object
        delta: Tolerance for deciding far field
        targets: Target vector
        p: Order of expansion
    """

    ff_potential = np.zeros(len(targets))  # far field potentials for each target

    for i, target in enumerate(targets):

        # far field potential
        u = 0.0

        q = deque([tree.root])

        while q:
            node = q.popleft()
            distance = abs(target - node.M)
            cell_size = node.U - node.L

            # if it is far field
            if distance > delta * cell_size:
                # compute phi and increment potential
                phi = eval_multipole_expansion(node, target, p)
                u += phi

            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

        # Store farfield potential
        ff_potential[i] = u

    return ff_potential


def eval_multipole_expansion(node: TreeNode, target: float, p: int):
    """Evaluate multipole expansion for a given target point

    Args:
        node: Current TreeNode for which the multipole expansion is evaluated
        target: The target point at which the potential is being calculated
        p: The order of the multipole expansion
    """

    phi = 0.0
    for m in range(p + 1):
        # S_m(x) = x^{-m}
        S_m = (node.M - target) ** (-m) if m > 0 else 1.0
        phi += node.weights[m] * S_m

    return phi


if __name__ == "__main__":

    # Create Tree
    T = (0, 1)
    k = 10
    tree = construct_tree(T, k)

    # Initiate sources and charges
    sources = np.array([0.1, 0.2, 0.6, 0.9])
    charges = np.array([1.0, -1.0, 1.0, -1.0])

    # order of expansion
    p = 2

    compute_weights_iterative(tree.root, sources, charges, p)

    targets = np.array([0.05, 0.25, 0.55, 0.85])

    far_field_potential = far_field_approximation(tree, targets, p)

    print("Far-Field Potential at targets:", far_field_potential)

    # tree.print_tree()
