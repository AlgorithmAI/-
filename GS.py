import numpy as np

def Gram_Schmidt(A):
    M = np.array(A,dtype='float64')
    m = M.shape[0]
    n = M.shape[1]
    #关闭科学计数法
    np.set_printoptions(suppress=True)
    # Q表示Q矩阵,R表示R矩阵且是方阵
    Q = np.zeros((m,n))
    #Q = np.zeros((m,n))
    R = np.zeros((n, n))
    cnt = 0

    for a in M.T:
        # a表示原矩阵的每一列的向量，为行向量
        q = np.copy(a)
        # 下面循环的作用，一个向量减去前面的向量的投影。再补充R矩阵对角线
        for i in range(0, cnt):
            # print("q1 = ", q)
            q -= Q[:, i] * (np.dot(Q[:, i].T, a))
            # print("q2 = ", q)
            # print("Q[:,i] = ", Q[:, i])
            # print("a = ", a)
            # print("m", np.dot(Q[:, i].T, a))
            R[cnt, i] = np.dot(Q[:, i].T, a)  # BUG:点积 = 0 而且R[cnt,i]=?<q,a>
        # print("qqq", q)
        # print("norm = ", np.linalg.norm(q))
        Q[:, cnt] = (q / np.linalg.norm(q))
        # print("qqq/norm1 = ", q / np.linalg.norm(q))
        # print("qqq/norm2 = ", Q[:, cnt])

        R[cnt, cnt] = np.linalg.norm(q)
        # print("Rduijiao", R[cnt, cnt])
        cnt += 1

    # for j in range(n):
    #     for i in range(m):
    #         if Q[i,j] > 1.7E-308:
    #             Q[i,j] = 0
    return [Q, R.T]

