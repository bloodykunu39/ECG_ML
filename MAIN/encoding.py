# module
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.polynomial.legendre import legendre
import math


def superposition(data:np.ndarray)->np.ndarray:
    """
    This function takes the data matrix and returns the superposition data matrix
    
    args:
    data : np.ndarray : data matrix of shape (5000, 12)

    returns:
    np.ndarray : superpositioned  img matrix of shape (5000, 5000)
    """
    scale=StandardScaler()
    # making each colum of 5000 datapoint should be in
    scaled = scale.fit_transform(data)# normal distribution of each column of the data matrix # feature scaling
    img=np.zeros((5000,5000))
    x = np.linspace(-1,1,5000)
    for i in range(data.shape[1]):
        leg=legendre(i+1)(x)
        norm=math.sqrt(np.sum(leg*leg))
        img=img+np.outer(scaled[:,i],#data is in shaPE OF 5000xDATA.shape[1] AN SHAPE[1] MEANS THE NUMBER OF COLUMNS
                         leg#legendre polinomial of order i+1 apllied to x p_n applies to matrix x
                         )/norm
    return img