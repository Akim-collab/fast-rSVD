import numpy as np

def eigsvd(A: np.ndarray):
    # function [U, S, V]= eigSVD(A)
    # A - (m,n), dense matrix, not sparce!
    # Computer economic SVD of matrix A (m>=n) using eig.
    m, n = A.shape
    B = np.dot(A.T, A)
    D, V = np.linalg.eig(B)
    S = np.diag(np.abs(D))
    U = np.dot(np.dot(A, V), np.linalg.inv(S))
    return U, S, V.T