import itertools
import math
import cmath
# mylist = list(itertools.chain.from_iterable(data))

import numpy as np
import pandas as pd

from numpy import random
# from random import random
import scipy
from scipy.spatial import distance
from scipy.stats import norm, beta

import matplotlib.pyplot as plt
from loading import loading

param = {'frustumLength': 20, 'frustumAperture': 160, 'frustumSamples': 2000, 'histnx': 20, 'histny': 20, 'sigma': 0.4,
         'method': 'JS', 'checkFacing': 1, 'HO_quantization': 1, 'FillMissDetection': 1, 'frustumMode': '',
         'checkOverlap': 0, 'weightMode': 'EQUAL', 'numFrames': 1, 'showWeights': 1, 'evalMethod': '', 'showgroups': 1,
         'showFrustum': 1, 'showResults': 1}

def frustumM(pos, orj, length, aperture, samples):
    pts = []
    ptsFM = np.zeros((samples, 2))
    leng = np.zeros((samples, 1))
    if length > 0:
        for i in range(0, samples):
            #sigma = ((aperture / 360 * (2 * math.pi))) / 3
            sigma = (1/3)*(aperture/(2*360*math.pi))
            orj = random.normal(orj, sigma)
            #leng = random.beta(0.8, 1.1, size=(2000,1)) * length
            leng = random.beta(0.8, 1.1)

            ptsFM[i, 0] = pos[0] + np.cos(orj) * leng * length
            ptsFM[i, 1] = pos[1] + np.sin(orj) * leng * length
    #print("ptsFM", ptsFM)
    return ptsFM


def hist2D(x, y, n_x, n_y, xrange, yrange):
    # tile repeat copies of array
    print("x before tile ", x)
    print(x.shape, "x.shape before")

    # x e y sono solo array, ovvero matrici con m righe e 1 colonna
    x = x - (np.tile(xrange[0], (x.shape[0], 1)))  # create a x.shape[0] by 1 copies of xrange[0]
    y = y - (np.tile(yrange[0], (y.shape[0], 1)))
    print("x", x)
    print(x.shape, "x.shape dopo")

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
    print("m", m)
    return m


def frustum():
    data = loading()
    persons = []
    for i in range(0, data.size):  # 100 array
        # ogni iterazione un array -> 0-99 iterazioni
        persons = data[0][i]
        # print(persons.shape)
        fxx = []

        if persons.size > 0:
            minx = math.inf
            miny = math.inf
            maxx = -math.inf
            maxy = -math.inf

            fxx = []
            for f in range(0, len(persons)):
                numberofcolors = persons.shape[0]
                # TODO trovare colori differenti
                listcolori = ['c', 'y', 'g', 'b', 'm', 'm', 'g', 'r', 'c', 'c', 'g', 'r']
                random_color = list(np.random.choice(range(255), size=2000))
                fx = frustumM([persons[f, 1], persons[f, 2]], persons[f, 3], param['frustumLength'],
                              param['frustumAperture'], param['frustumSamples'])
                plt.scatter(fx[:, 0], fx[:, 1], marker='o', c=listcolori[f])  # "#1E8D5C")#np.random.rand(2000))
                fxx.append(fx)
            #plt.show()
            print("fxxx", len(fxx))
            #ciclo fxx e salvo in una variabile temporanea

            for n in range(0, persons.shape[0]):
                for m in range(0, len(fxx)):
                    if minx > fxx[n][m][0]: minx = fxx[n][m][0]
                    if miny > fxx[n][m][1]: miny = fxx[n][m][1]
                    if maxx < fxx[n][m][0]: maxx = fxx[n][m][0]
                    if maxy < fxx[n][m][1]: maxy = fxx[n][m][1]


            px = np.zeros(shape=persons.shape[0])
            hx = np.zeros(shape=(param['histnx'], param['histny']))
            #hx = np.zeros(shape=persons.shape[0])
            pxhist = np.zeros(shape=(persons.shape[0],400))

            for h in range(0, persons.shape[0]):
                # get the 2D histogram for each person  
                hx = hist2D(fxx[h][:,0], fxx[h][:,1], param['histnx'], param['histny'], [minx, maxx], [miny, maxy])

                #print("hx", hx)
                #print("hx.shape", hx.shape)
'''
                # create a row vector of the histogram - linearize
                px = np.reshape(a=(hx), newshape=(param['histnx']* param['histny']))

                for j in range(0, np.size(px)):
                    pxhist[i][j] = px[j]

                #plt.hist(pxhist[i,:], bins = 100) # appare solo la colonna
                #plt.show()

            #print("la prima riga", pxhist[0,:])
            
'''
'''
            # evaluate pairwise affinity matrix
            if np.size(pxhist) > 1:
                #matrice quadrata numero persone * numero persone
                distmat = np.zeros(shape=(pxhist.shape[0], pxhist.shape[0])) 
                if param['method'] == 'JS':
                    for k in range(0, 5): 
                        for i in range(0, pxhist.shape[0]): 
                             for j in range(i, pxhist.shape[0]):                
                                distmat[i][j] = distance.jensenshannon(p=pxhist[i,:], q=pxhist[j,:])**2
                                #distmat[i][j] =js(p=pxhist[i,:], q=pxhist[j,:])
                                distmat[j][i] = distmat[i][j]

            #print("distmat", distmat)

            #m = not (math.isinf(distmat))

            affinitymat = np.zeros(shape=(distmat.shape[0], distmat.shape[0]))
            for i in range(0, distmat.shape[0]):
                for j in range(0, distmat.shape[0]):
                    affinitymat[i][j] = math.exp(-distmat[i][j] / param['sigma'])# * (not (np.eye(distmat.shape[0], distmat.shape[1])))

            #affinitymat = affinitymat * m
            #print("affinitymat", affinitymat)
'''
