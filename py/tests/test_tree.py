import numpy as np
import sys
import unittest

sys.path.append("..")
import tree as t


class TestMultipoleWeights(unittest.TestCase):

    def test_multipole_weights(self):

        # Create Tree
        T = (0, 1)
        k = 3
        p = 2
        tree = t.construct_tree(T, k)

        # initialize sources and charges
        sources = np.array([0.1, 0.2, 0.6, 0.9])
        charges = np.array([1.0, -1.0, 1.0, -1.0])

        # Compute weights iterative using level order
        t.compute_weights_iterative(tree.root, sources, charges, p)

        # Brute-force reference computation

        # Expected weights for node 1
        expected_weights = np.array(
            [
                np.sum(charges),
                np.sum(charges * (sources - tree.root.M)),
                np.sum(charges * (sources - tree.root.M) ** 2),
            ]
        )

        self.assertTrue(np.allclose(tree.root.weights, expected_weights, atol=1e-6))


if __name__ == "__main__":

    unittest.main()
