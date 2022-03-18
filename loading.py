import numpy as np
from scipy.io import loadmat


def loading():
    data = loadmat('data/Synth/features.mat')
    new_list = data['features']#.flatten()

    return new_list
    #print(new_list.shape)
    #print(new_list)
