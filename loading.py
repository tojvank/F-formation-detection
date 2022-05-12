import numpy as np
from scipy.io import loadmat


def loading():
    data = loadmat('data/Synth/features.mat')
    groundTruth = loadmat('data/Synth/groundtruth.mat')

    data1 = loadmat('data/CocktailParty/features.mat')
    groundTruth1 = loadmat('data/CocktailParty/groundtruth.mat')
    data2=loadmat('data/PosterData/features.mat')
    data3 =loadmat('data/Savarese/features.mat')
    savareseDb = data3['features']

    synthDb = data['features']#.flatten()
    cocktailDb = data1['features']
    posterDataDb =data2['features']
    # sprint("shape savareseDb",savareseDb.shape)
    # print(savareseDb)
    return savareseDb



