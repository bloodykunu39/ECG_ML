# module
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.polynomial.legendre import legendre
import math


def smooth(data:np.ndarray)->np.ndarray:
    """
    This function takes the data matrix and returns the smoothened data matrix
    
    args:
    data : np.ndarray : data matrix of shape (5000, 12)

    returns:
    np.ndarray : smoothened data matrix of shape (5000, 5000)
    """
    scale=StandardScaler()
    # making each colum of 5000 datapoint should be in
    scaled = scale.fit_transform(data)# normal distribution of each column of the data matrix # feature scaling
    sum=np.zeros((5000,5000))
    x = np.linspace(-1,1,5000)
    for i in range(data.shape[1]):
        leg=legendre(i+1)(x)
        norm=math.sqrt(np.sum(leg*leg))
        sum=sum+np.outer(scaled[:,i],#data is in shaPE OF 5000xDATA.shape[1] AN SHAPE[1] MEANS THE NUMBER OF COLUMNS
                         leg#legendre polinomial of order i+1 apllied to x p_n applies to matrix x
                         )/norm
    return sum

def crossgrain(data:np.ndarray,cg:int=50)->np.ndarray:
    """
    This function takes the data matrix and returns the coarse grained data matrix to the given factor cg

    args:
    data : np.ndarray : data matrix of shape (5000, 12)
    cg : int : factor to coarse grain the data matrix

    returns:
    np.ndarray : coarse grained data matrix of shape (5000/cg,5000/cg)

    """
    img = smooth(data)#superposition of the data matrix
    
    s1=np.zeros((int(img.shape[0]/cg)#shape[0] is the number of rows
                 ,int(img.shape[1]/cg))#shape[1] is the number of columns
                 ) # Coarse grained image'

    for i in range(s1.shape[0]):# s1 is zero matrix of shape[0] rows and shape[1] columns
        for j in range(s1.shape[1]):
            s1[i,j]=np.mean(img[i*cg:i*cg+cg,j*cg:j*cg+cg]) # changing the value of the s1 zero matrix to the mean of the fxf block of the img matrix

    return s1
