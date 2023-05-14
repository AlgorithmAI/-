import numpy as np
from LU import LU_Dec

def determinant(A):#A表示输入的矩阵
    M = np.array(A,dtype='float64')
    #m,n是矩阵的行数和列数
    m = M.shape[0]
    n = M.shape[1]

    if m != n:
        return "不是方阵，没有对应的行列式"
    elif np.linalg.matrix_rank(M) != m:
        return 0
    else:
        #K是A经过LU分解后的列表，第一个元素是P，第二个元素是L，第三个矩阵是U
        K = LU_Dec(A)
        U = K[2]
        deter = U[0,0]
        for i in range(1,m):
            deter *= U[i,i]
        deter = deter*np.linalg.det(K[0])

        return deter

# A = [[2,2,2],[4,7,7],[6,18,22]]
# k = determinant(A)
# print(k)