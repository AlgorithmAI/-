import  numpy as np

def Householder_Reduction(A):
    np.set_printoptions(suppress=True)
    M = np.array(A, dtype='float64')
    m = M.shape[0]
    n = M.shape[1]

    P = np.eye(m, dtype='float64')

    for j in range(min(m-1,n)):
        I = np.eye(m, dtype='float64')
        # 判断此列主对角线下面是否全是0
        Norm_condition = np.array(M[j + 1:m, j],dtype='int32')
        # print("Norm_condition = ",Norm_condition)
        if np.all(Norm_condition == 0):
            continue
        else:
            e = np.zeros(m - j, dtype='float64')
            e[0] = 1

            u = M[j:m, j] - np.linalg.norm(M[j:m, j]) * e#一会改为减法！！！！！！！
            u_u = np.array([u])

            u_t = u.reshape(1, -1).T
            dot_c = np.dot(u_t, u_u)
            di = np.dot(u,u)
            I_temp = -2 /di*dot_c
            row, col = np.diag_indices_from(I_temp)
            I_temp[row, col] += 1#I_temp = R1 R2.....
            I[j:m, j:m] = I_temp

            P = np.dot(I, P)
            M = np.dot(I, M)

    return [P.T, M]
# A = [[-4,-2,-4,-2],[2,-2,2,1],[-4,1,-4,-2]]
# k = Householder_Reduction(A)
# # print(k)