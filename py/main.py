import numpy as np


# summation formula for a given function


# for a phi where phi is = (x[i] - y[j]**2)


# x and y are just a list of points


def summation(phi: np.array, q: np.array):

    N = len(phi[0])

    u = np.zeros(N, 1)

    for j in range(1, N):
        alpha = alpha + q[j]
        beta = alpha + q[j] * x[j]
        gamma = gamma + q[j] * (x[j] ** 2)

    for i in range(1, N):
        u[i] = alpha * y[i] ** 2 - 2 * beta * y[i] + gamma

    return u


def main():

    print("Hello from fmm!")


if __name__ == "__main__":
    main()
