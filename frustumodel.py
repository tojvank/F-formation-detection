import itertools
import math
import cmath
# mylist = list(itertools.chain.from_iterable(data))

import numpy as np
import pandas as pd

from numpy import random
#from random import random
import scipy
from scipy.spatial import distance
from scipy.stats import norm, beta

import matplotlib.pyplot as plt
from loading import loading



def frustumM(pos, orj, length, aperture, samples):
    pts = []
    ptsFM = np.zeros((samples, 2))
    aperture = aperture / 2
    if length > 0:
        for i in range(0, samples):
            sigma = ((aperture / 360 * (2 * math.pi))) / 3
            orj = random.normal(orj, sigma)
            leng = random.beta(0.8, 1.1) * length

            ptsFM[i, 0] = pos[0] + np.cos(orj) * leng
            ptsFM[i, 1] = pos[1] + np.sin(orj) * leng
    return ptsFM


def frustum():

    data = loading()
    persons = []

    for m in range(0, 8):#data.size):
        persons.append(data[0][m])


    print(persons[0])
    print(len(persons)) # sarebbero anche il numero di persone esistenti in quel gruppo
    
    fxx = []
    for f in range(0, len(persons)):
    #fx = frustumM(persons[f][1], persons[f][2], persons[f][3], 20,160,2000)
     ''' 

        if persons.size != 0:

            minx = math.inf
            miny = math.inf
            maxx = -math.inf
            maxy = -math.inf

            persons = data[0][m]
            
            fxx = []

            for f in range(0, persons.shape[0]):
                #fx = frustumM([persons[f, 1], persons[f, 2]], persons[f, 3],param['frustumLength'], param['frustumAperture'], param['frustumSamples'])
                fx = frustumM([persons[f, 1], persons[f, 2]], persons[f, 3], 20, 160, 2000)

                #fx = frustumM([59,32], 0.52, 20, 160, 2000)
                print("fx", fx)
                #fxx.append(fx)
                print("fxx", fxx)
                print("fxx.shape", len(fxx))
                print("fx", fx.shape)
            return fxx
            '''

    '''
                plt.scatter(fx[:, 0], fx[:, 1], c=np.random.rand(2000))
            plt.show()
            
    '''
    '''            
            for f in range(0, persons.shape[0]):            
                fx = frustumM([persons[f, 1], persons[f, 2]], persons[f, 3],
                                                    param['frustumLength'], param['frustumAperture'],
                                                    param['frustumSamples'])
                fxx.append(fx)
                print("fx", fx.shape)

                plt.scatter(fx[:,0], fx[:,1], c=np.random.rand(1))
                plt.show()


                fxx[f][0] = fx[0]
                fxx[f][1] = fx[1]

                if minx > np.min(fxx[:,0]):
                    minx=np.min(fxx[:,0])
                if miny > np.min(fxx[:,1]):
                    miny=np.min(fxx[:,1])
                if maxx > np.max(fxx[:,0]):
                    maxx=np.max(fxx[:,0])
                if maxy > np.max(fxx[:,1]):
                    maxy=np.max(fxx[:,1])

            #scatterplot dei frustum  
            #print("Frame n.", m)
            #plt.scatter(fxx[:,0], fxx[:,1])
            #plt.show()

            px = np.zeros(shape=persons.shape[0])
            hx = np.zeros(shape=(param['histnx'], param['histny']))
            #hx = np.zeros(shape=persons.shape[0])
            pxhist = np.zeros(shape=(persons.shape[0],400))

            for i in range(0, persons.shape[0]):            
                # get the 2D histogram for each person  
                hx = hist2D(fxx[:,0], fxx[:,1], param['histnx'], param['histny'],[minx, maxx], [miny, maxy])

                #print("hx", hx)
                #print("hx.shape", hx.shape)
                # create a row vector of the histogram - linearize
                px = np.reshape(a=(hx), newshape=(param['histnx']* param['histny']))

                for j in range(0, np.size(px)):
                    pxhist[i][j] = px[j]

                #plt.hist(pxhist[i,:], bins = 100) # appare solo la colonna
                #plt.show()

            #print("la prima riga", pxhist[0,:])
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





