
import math
from random import random
import numpy as np
import scipy
import scipy as scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import math
import cmath
from numpy import random
import scipy
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.spatial import distance

import matplotlib.pyplot as plt
from scipy.stats import norm, beta
# x and y are vectors of mx1 positions in the x directions and y directions respectively
def hist2d(x, y, n_x, n_y, xrange, yrange):
    # generate a 2D histogram of a given set of points, the histogram is normalized so it adds up to 1
    # tile repeat copies of array
    # print("x before tile ", x)
    # print(x.shape, "x.shape before")

    # x e y sono solo array, ovvero matrici con m righe e 1 colonna
    x = x - (np.tile(xrange[0], (x.shape[0], 1)))  # create a x.shape[0] by 1 copies of xrange[0]
    y = y - (np.tile(yrange[0], (y.shape[0], 1)))
    # print("x", x)
    # print(x.shape, "x.shape")

    m = np.zeros((n_x, n_y))  # 20 x 20

    stepx = (xrange[1] - xrange[0]) / n_x
    stepy = (yrange[1] - yrange[0]) / n_y
    # print("stepx", stepx, "stepy", stepy)
    # LINEARIZZO X E Y
    linx = np.zeros((np.size(x)))
    liny = np.zeros((np.size(y)))
    for k in range(0, np.size(x)):
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[0]):
                linx[k] = x[i][j]
                liny[k] = y[i][j]

    # print("linxx", linx)
    # print("linyy", liny)
    for j in range(0, np.size(linx)):
        # print("linx", linx[j])
        xx = math.ceil(linx[j] / stepx)
        yy = math.ceil(liny[j] / stepy)
        # print("xx", xx, "yy", yy)

        if xx == 0:
            xx = 1
        if xx > n_x:
            xx = n_x
        if yy == 0:
            yy = 1
        if yy > n_y:
            yy = n_y

        # posizioni partono da 0 ed arrivano a 19, quindi ovviamente mi dà errore su 20
        m[yy][xx] = m[yy][xx] + 1

    m = m / np.size(linx)
    # print("m", m)
    return m
