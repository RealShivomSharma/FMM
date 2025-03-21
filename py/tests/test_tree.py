import numpy as np
import sys
import pytest

sys.path.append("..")
import tree as t


def test_multipole_weights():

    # Create Tree
    T = (0, 1)
    k = 20
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

    assert np.allclose(tree.root.weights, expected_weights, atol=1e-6)


# TODO: Need to most likely fix this test/review FFA
def test_far_field_potential():
    T = (0, 1)
    k = 15
    p = 2
    delta = 2.0
    tree = t.construct_tree(T, k)

    sources = np.array([0.1, 0.2, 0.6, 0.9])
    charges = np.array([1.0, -1.0, 1.0, -1.0])

    t.compute_weights_iterative(tree.root, sources, charges, p)

    targets = np.array([0.05, 0.25, 0.55, 0.85])
    far_field_potential = t.far_field_approximation(tree, delta, targets, p)

    # Brute-force direct computation
    direct_potential = np.array(
        [np.sum(charges / np.abs(target - sources)) for target in targets]
    )

    assert np.allclose(far_field_potential, direct_potential, atol=1e-3)
