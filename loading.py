import numpy as np
from scipy.io import loadmat


def loading():
    data = loadmat('data/Synth/features.mat')
    groundTruth = loadmat('data/Synth/groundtruth.mat')

    data1 = loadmat('data/CocktailParty/features.mat')
    groundTruth1 = loadmat('data/CocktailParty/groundtruth.mat')

    synthDb = data['features']#.flatten()
    cocktailDb = data1['features']

    #print(cocktailDb.shape)
    #print(cocktailDb)
    return cocktailDb



