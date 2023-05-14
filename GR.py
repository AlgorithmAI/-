import numpy as np

def Givens_Reduction(A):
    np.set_printoptions(suppress=True)
    m = np.array(A).shape[0]
    n = np.array(A).shape[1]

    #Q是吉文斯矩阵连乘且是方阵，R是上三角矩阵。PA=R => P.T=Q
    P = np.eye(m,dtype='float64')
    #R = np.zeros((m,n))

    M = np.array(A,dtype = 'float64')
    #
    for j in range(min(m-1,n)):
        for i in range(j+1,m):
            M_temp = np.array(M[i,j],dtype='int32')
            if M_temp == 0:
                continue
            else:
                #Pji
                P_temp = np.eye(m,dtype='float64')
                CADS_temp = (M[j,j]**2 + M[i,j]**2)**0.5
                P_temp[j,j] = M[j,j]/CADS_temp
                P_temp[i,i] = P_temp[j,j]
                P_temp[j,i] = M[i,j]/CADS_temp
                P_temp[i,j] = -P_temp[j,i]
                #print("P = ",j+1,i+1,P_temp)
                P = np.dot(P_temp,P)
                M = np.dot(P_temp,M)
                #print("PA = ",M)

    return [P.T,M]
