# module
import numpy as np
from sklearn.preprocessing import StandardScaler
import math


def coarsegrain(img: np.ndarray ,cg:int=50)->np.ndarray:
    """
    This function takes the data matrix and returns the coarse grained data matrix to the given factor cg

    args:
    img : np.ndarray : superpostion matrix of shape (5000, 5000)
    cg : int : factor to coarse grain the data matrix

    returns:
    np.ndarray : coarse grained data matrix of shape (5000/cg,5000/cg)

    """
        
    s1=np.zeros((int(img.shape[0]/cg)#shape[0] is the number of rows
                 ,int(img.shape[1]/cg))#shape[1] is the number of columns
                 ) # Coarse grained image'

    for i in range(s1.shape[0]):# s1 is zero matrix of shape[0] rows and shape[1] columns
        for j in range(s1.shape[1]):
            s1[i,j]=np.mean(img[i*cg:i*cg+cg,j*cg:j*cg+cg]) # changing the value of the s1 zero matrix to the mean of the fxf block of the img matrix

    return s1
