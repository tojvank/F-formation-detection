import numpy as np
from scipy.io import loadmat


def loading():
    data = loadmat('data/Synth/features.mat')
    new_list = data['features']#.flatten()

    return new_list
    #print(new_list.shape)
    #print(new_list)

'''

def frustumMa(pos, orj, length, aperture, samples):
    pts = []
    ptsFM = []
    aperture = aperture/2
    if length > 0:
        for i in range(0, samples):
            sigma = ((aperture/360 *(2*math.pi)))/3
            orj = norm.rvs(orj, sigma)
            len = beta.rvs(0.8, 1.1) * length
            #len = np.sort(len)
            ptsFM = ptsFM.append((pos[0]+np.cos(orj)*len,pos[1]+np.sin(orj)*len))
            return ptsFM

'''
'''
                if minx > np.min(fxx[:, 0]):
                    minx = np.min(fxx[:, 0])
                if miny > np.min(fxx[:, 1]):
                    miny = np.min(fxx[:, 1])
                if maxx > np.max(fxx[:, 0]):
                    maxx = np.max(fxx[:, 0])
                if maxy > np.max(fxx[:, 1]):
                    maxy = np.max(fxx[:, 1])
'''

