from frustumodel import frustum, frustumM

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from ds import *

import pandas as pd

param = {'frustumLength': 20, 'frustumAperture': 160, 'frustumSamples': 2000, 'histnx': 20, 'histny': 20, 'sigma': 0.4, 'method': 'JS', 'checkFacing': 1, 'HO_quantization': 1, 'FillMissDetection': 1,'frustumMode': '', 'checkOverlap': 0, 'weightMode': 'EQUAL', 'numFrames': 1, 'showWeights': 1, 'evalMethod': '', 'showgroups': 1, 'showFrustum': 1, 'showResults': 1}

if __name__ == '__main__':

    frustum()
    


'''
if __name__ == "__main__":

    np.random.seed(1)

    n = 1000
    d = 2

    X, y = make_blobs(n, d, centers=3)

    D = pairwise_distances(X, metric='sqeuclidean')

    sigma2 = np.median(D)

    S = np.exp(-D / sigma2)

    x = dominant_set(S, epsilon=2e-4)
    
    if d == 2:
        plt.figure()
        for yi in np.unique(y):
            plt.plot(X[y == yi, 0], X[y == yi, 1], 'o')

        plt.title('Dataset')
    
    plt.figure()
    plt.imshow(S, interpolation='nearest')
    plt.title('similarity matrix')

    idx = np.argsort(x)[::-1]
    B = S[idx, :][:, idx]
    plt.figure()
    plt.imshow(B, interpolation='nearest')
    plt.title('Re-arranged similarity matrix')
    plt.figure()
    plt.semilogy(np.sort(x))
    plt.title('Sorted weighted characteristic vector (x)')

    cutoff = np.median(x[x > 0])
    print("cutoff:", cutoff)
    plt.figure()
    plt.plot(X[x <= cutoff, 0], X[x <= cutoff, 1], 'bo')
    plt.plot(X[x > cutoff, 0], X[x > cutoff, 1], 'ro')
    plt.title("Dominant set")

    plt.show()
    
    DF_var = pd.DataFrame.from_dict({"s1": [1.2, 3.4, 10.2],
                                     "s2": [1.4, 3.1, 10.7],
                                     "s3": [2.1, 3.7, 11.3],
                                     "s4": [1.5, 3.2, 10.9]})
    DF_var.index = ["g1", "g2", "g3"]
    mat = squareform(pdist(DF_var, metric='euclidean'))
    print(mat)
    mat1 = dominant_set(mat, epsilon=2e-4)

    Z, k = make_blobs(1000, d, centers=3)

    cutoff = np.median(mat1[mat1 > 0])
    print("cutoff:", cutoff)
    plt.figure()
    plt.plot(Z[mat1 <= cutoff, 0], Z[mat1 <= cutoff, 1], 'bo')
    plt.plot(Z[mat1 > cutoff, 0], Z[mat1 > cutoff, 1], 'ro')
    plt.title("Dominant set")
    plt.show()
    
'''

