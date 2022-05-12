import numpy as np
from sklearn.cluster import DBSCAN

from frustumodel import frustum
from loading import loading

def dbscanClustering():
    counter = loading()
    counter = counter.shape[1]
    print("counter", counter)
    for i in range(0, counter):
        distmat = frustum()
        print("distmat dbscan", distmat)
        clustering = DBSCAN(eps=4, min_samples=5).fit(distmat)
        clustering.labels_
        print("labels", clustering.labels_)