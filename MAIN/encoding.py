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
    img=np.zeros((n,n))
    x = np.linspace(-1,1,n)
    for i in range(data.shape[1]):
        leg=legendre(i+1)(x)
        norm=math.sqrt(np.sum(leg*leg))
        img=img+np.outer(scaled[:,i],#data is in shaPE OF 5000xDATA.shape[1] AN SHAPE[1] MEANS THE NUMBER OF COLUMNS
                         leg#legendre polinomial of order i+1 apllied to x p_n applies to matrix x
                         )/norm
    return img

def inverse_superposition(img: np.ndarray, num_leads: int = 12) -> np.ndarray:
    """
    Reconstruct lead signals from the superposition image.

    args:
    img : np.ndarray : superposition image matrix of shape (n, n)
    num_leads : int : number of ECG leads to reconstruct (default=12)

    returns:
    np.ndarray : reconstructed lead matrix of shape (n, num_leads)
    """
    n = img.shape[0]
    x = np.linspace(-1, 1, n)
    inversed = []

    for i in range(1, num_leads + 1):
        leg = legendre(i)(x)
        norm = math.sqrt(np.sum(leg * leg))
        inversed.append(np.dot(img, leg) / norm)

    return np.array(inversed).T