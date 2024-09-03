# module
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import legendre
import math
def normalize_matrix(matrix):
    """take input as 500*12 or a person report and return the normalized matrix 
    >>>sum of square of all the elements of the norm_matrix is 1"""
    
    matrix_norm = np.linalg.norm(matrix, 'fro')#frobenius norm is the square root of the sum of the absolute squares of its elements
                                                # works for 2d arrays
    if np.isnan(matrix_norm):# is NAN not a number 
        #it will return a matrix of zeros of the same shape as the input matrix
        return np.zeros_like(matrix)
    else:
        normalized_matrix = matrix / matrix_norm
        return normalized_matrix

def superposition(data:np.ndarray,n)->np.ndarray:
    """
    This function takes the data matrix and returns the superposition data matrix
    
    args:
    data : np.ndarray : data matrix of shape (5000, 12)

    returns:
    np.ndarray : superpositioned  img matrix of shape (5000, 5000)
    """
    
    scale=StandardScaler()
    # making each colum of 5000 datapoint should be in
    scaled = scale.fit_transform(normalize_matrix(data))# normal distribution of each column of the data matrix # feature scaling
    img=np.zeros((5000,5000))
    x = np.linspace(-1,1,n)
    for i in range(data.shape[1]):
        leg=legendre(i+1)(x)
        norm=math.sqrt(np.sum(leg*leg))
        img=img+np.outer(scaled[:,i],#data is in shaPE OF 5000xDATA.shape[1] AN SHAPE[1] MEANS THE NUMBER OF COLUMNS
                         leg#legendre polinomial of order i+1 apllied to x p_n applies to matrix x
                         )/norm
    return img