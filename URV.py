import numpy as np

from GR import Givens_Reduction
from HR import Householder_Reduction

def URV_Factorization(A):
    np.set_printoptions(suppress=True)
    # P = Householder_Reduction(A)[0].T
    # B = Householder_Reduction(A)[1]    #PA = B

    P = Givens_Reduction(A)[0].T
    B = Givens_Reduction(A)[1]

    m = B.shape[0]
    n = B.shape[1]
    r = np.linalg.matrix_rank(B)

    #B_temp = (B[0:r].T).tolist()
    # Q = Householder_Reduction(B[0:r].T)[0].T
    Q = Givens_Reduction(B[0:r].T)[0].T
    T_temp = Householder_Reduction(B[0:r].T)[1][0:r,0:r].T
    #T_temp = Givens_Reduction(B[0:r].T)[1][0:r,0:r].T

    # T = np.zeros((m,n),dtype='float64')
    # T[0:r,0:r] = T_temp
    R = np.zeros((m,n),dtype='float64')
    R[0:r,0:r] = T_temp

    U = P.T
    V = Q

    return [U,R,V]

# A = [[-4,-2,-4,-2],[2,-2,2,1],[-4,1,-4,-2]]
# URV_Factorization(A)
# print(np.all([0.,0.,-0.] == 0))
# print("U = \n",URV_Factorization(A)[0],"\n",
#       "R = \n",URV_Factorization(A)[1],"\n",
#       "V.t = \n",URV_Factorization(A)[2],
#       "   =",np.dot(URV_Factorization(A)[2],np.array([[6,0],[0,3],[6,0],[3,0]]))
#
#     )