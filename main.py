from frustumodel import frustum, frustumM

'''
param = {'frustumLength': 20, 'frustumAperture': 160, 'frustumSamples': 2000, 'histnx': 20, 'histny': 20, 'sigma': 0.4, 'method': 'JS', 'checkFacing': 1, 'HO_quantization': 1, 'FillMissDetection': 1,'frustumMode': '', 'checkOverlap': 0, 'weightMode': 'EQUAL', 'numFrames': 1, 'showWeights': 1, 'evalMethod': '', 'showgroups': 1, 'showFrustum': 1, 'showResults': 1}

if __name__ == '__main__':

    frustum()
    
'''


if __name__ == "__main__":

    from sklearn.metrics import pairwise_distances
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

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
    print
    "cutoff:", cutoff
    plt.figure()
    plt.plot(X[x <= cutoff, 0], X[x <= cutoff, 1], 'bo')
    plt.plot(X[x > cutoff, 0], X[x > cutoff, 1], 'ro')
    plt.title("Dominant set")

    plt.show()