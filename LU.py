import numpy as np
import copy
from Materix_Row_Transformation import Matrix_Row_Transformation

def LU_Dec(A):  # 部分主元法的LU分解，返回P、L、U矩阵
    # m是行数，n是列数
    m = np.array(A).shape[0]
    n = np.array(A).shape[1]
    np.set_printoptions(suppress=True)
    if (m != n):
        return "The Matrix is not square!Can't use LU decomposition!"
    elif (np.linalg.det(A) == 0):
        return "The Matrix is singular！Can't use LU decomposition!"
    else:
        # L表示LU分解的L矩阵
        P_Location = list(range(1, n + 1))  # 记录行交换顺序
        L = np.zeros((n, n))
        R_T = Matrix_Row_Transformation()
        R_T.A = A
        L_T = Matrix_Row_Transformation()
        L_T.A = L

        for j in range(n):  # 对A矩阵的列
            # print(j)
            Pivot_Elem_temp = copy.deepcopy(R_T.A[j][j])
            Row_Max_Num = j

            # 主对角线下面的元素,得到主元位置的行索引
            for i in range(j + 1, n):
                if abs(Pivot_Elem_temp) >= abs(R_T.A[i][j]):
                    continue
                else:
                    Row_Max_Num = i  # 主元所在行的行索引
                    Pivot_Elem_temp = copy.deepcopy(R_T.A[i][j])
            # 在P_Location中记录行交换的位置
            temp = copy.deepcopy(P_Location[Row_Max_Num])
            P_Location[Row_Max_Num] = P_Location[j]
            P_Location[j] = temp
            R_T.Two_Row_Exchange(Row_Max_Num + 1, j + 1)
            L_T.Two_Row_Exchange(Row_Max_Num + 1, j + 1)
            # 消掉主元下面的元素，U矩阵主对角线下面记录Lij
            for k in range(j + 1, n):
                if R_T.A[k][j] == 0:
                    continue

                else:
                    L_T.A[k][j] = copy.deepcopy(R_T.A[k][j] / Pivot_Elem_temp)
                    R_T.One_Add_Kone(j + 1, k + 1, -(R_T.A[k][j] / Pivot_Elem_temp))

        # 构造P矩阵
        P = np.zeros((n, n))
        for j in range(n):
            P[j][P_Location[j] - 1] = 1

        # 改造L矩阵
        for j in range(n):
            L_T.A[j][j] = 1
            # R_T.A[i][j] = 0

        P_arr = np.array(P)
        L_arr = np.array(L_T.A, dtype='float64')
        U_arr = np.array(R_T.A, dtype='float64')

    return [P_arr,L_arr,U_arr]  # 此处返回P、L、
A = [[2,2,2],[4,7,7],[6,18,22]]
print(LU_Dec(A))