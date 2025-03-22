import numpy as np
from scipy.linalg import svd


class MatrixNode:

    # Lower and Upper bound
    def __init__(self, row_start, row_end, col_start, col_end):

        self.row_range = (row_start, row_end)
        self.col_range = (col_start, col_end)
        self.children = []
        self.data = None
        self.is_leaf = False  # Indicator of leaf node

    @property
    def shape(self):
        r_start, r_end = self.row_range
        c_start, c_end = self.col_range
        return (r_end - r_start, c_end - c_start)


class HMatrix:
    """
    Class to convert a dense matrix into it's hierarchical structure (Analogous to the interval divisions for sources and charges)
    Rather than using a binary tree as in the field case, Matrix Multiplication requires a QuadTree structure
    We will use this property to subdivide the matrix into blocks
    """

    def __init__(self, matrix: np.array, max_rank=5, min_size=16, tol=1e-6):
        self.matrix = matrix
        self.nrows, self.ncols = matrix.shape
        self.root = MatrixNode(0, self.nrows, 0, self.ncols)
        self.max_rank = max_rank
        self.min_size = min_size
        self.tol = tol

        # Construct tree from root matrix
        self.construct_tree(self.root, matrix)

    def construct_tree(self, node: MatrixNode, sub_matrix: np.array):

        r_start, r_end = node.row_range
        c_start, c_end = node.col_range

        # If the size is too small end recursion

        if node.shape[0] <= self.min_size or node.shape[1] <= self.min_size:
            node.is_leaf = True  # no longer subdividing
            U, s, Vh = svd(sub_matrix, full_matrices=False)
            rank = np.sum(s > self.tol)

            if rank <= self.max_rank:
                # If rank is low enough then we perform and store low rank approximation
                node.data = (U[:, :rank] * s[:rank], Vh[:rank, :])
            else:
                # Otherwise keep dense structure
                node.data = sub_matrix.copy()

            return

        r_mid = (r_start + r_end) // 2
        c_mid = (c_start + c_end) // 2

        partitions = [
            (r_start, r_mid, c_start, c_mid),  # Top left
            (r_start, r_mid, c_mid, c_end),  # Top Right
            (r_mid, r_end, c_start, c_mid),  # Bottom Left
            (r_mid, r_end, c_mid, c_end),  # Bottom Right
        ]
        for r_start, r_end, c_start, c_end in partitions:
            child = MatrixNode(r_start, r_end, c_start, c_end)
            node.children.append(child)
            self.construct_tree(child, self.matrix[r_start:r_end, c_start, c_end])


if __name__ == "__main__":

    A = np.random.rand(4, 4)

    test_matrix = HMatrix(A)

    print(test_matrix.root.data)
