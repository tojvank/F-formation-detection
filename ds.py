import numpy as np
from numpy.linalg import norm


def dominant_set(A, x=None, epsilon=1.0e-4):
    """Compute the dominant set of the similarity matrix A with the

    . Convergence is reached
    when x changes less than epsilon.
    """
    if x is None:
        x = np.ones(A.shape[0]) / float(A.shape[0])

    distance = epsilon * 2
    while distance > epsilon:
        x_old = x.copy()
        # x = x * np.dot(A, x) # this works only for dense A
        x = x * A.dot(x)  # this works both for dense and sparse A
        x = x / x.sum()
        distance = norm(x - x_old)
        print(x.size, distance)

    return x

