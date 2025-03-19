from collections import deque


class TreeNode:

    # Lower and Upper bound
    def __init__(self, L, U):
        self.L = L
        self.U = U
        self.left = None
        self.right = None


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
                    nodes.append([node.L, node.U])
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


if __name__ == "__main__":
    T = (0, 1)
    k = 3
    tree = construct_tree(T, k)
    tree.print_tree()
# Construction algorithm
