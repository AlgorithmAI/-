import numpy as np
from URV import URV_Factorization
from LU import LU_Dec
from HR import Householder_Reduction
from GS import Gram_Schmidt
from GR import Givens_Reduction


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

def Solu(A,b,string):

    #URV分解
    if string == "URV":
        m = np.array(A).shape[0]
        n = np.array(A).shape[1]

        B = np.array([b]).T
        SV = URV_Factorization(A)
        U = SV[0]
        R = SV[1]
        V = SV[2]

        r = np.linalg.matrix_rank(SV[1])
        T = SV[1][0:r,0:r]
        R_t = np.linalg.inv(T)
        R_temp = np.zeros((n,m))
        R_temp[0:r,0:r] = R_t


        A_n = np.dot(V.T,R_temp)
        Ani = np.dot(A_n,U.T)
        x = np.dot(Ani,B)
        return [U,R,V,x]
    #LU分解,A是方阵
    elif string == "LU":
        B = np.array([b]).T
        m = np.array(A).shape[0]
        Y = np.zeros((m,1))
        X = np.zeros((m,1))
        if B.shape[0] != m:
            return "Input Error!"
        else:
            PLU = LU_Dec(A)
            Pb = np.dot(PLU[0],B)
            L = PLU[1]
            U = PLU[2]

            #LUx=b  LY=b,解Y
            Y[0] = Pb[0]
            for i in range(1,m):
                sum1 = 0
                for k in range(i):
                    sum1 += L[i,k]*Y[k]
                Y[i] = Pb[i]-sum1

            #Ux = Y,解x
            X[m-1] = Y[m-1]/U[m-1,m-1]
            for i in range(m-2,-1,-1):
                sum2 = 0
                for k in range(i+1,m):
                    sum2 += U[i,k]*X[k]
                X[i] = (Y[i]-sum2)/U[i,i]

            return [PLU[0],L,U,X]
    #Householder
    elif string == "HR":
        B = np.array([b]).T
        # m = np.array(A).shape[0]
        # n = np.array(A).shape[1]
        H = Householder_Reduction(A)
        Q_Tb = np.dot(H[0].T,B)
        return [H[0],H[1],np.linalg.solve(H[1],Q_Tb)]
        # r = min(m,n)
        # R = Q_Tb[0:r,0:r]
#   #吉文斯约简(A)
    elif string == "GR":
        B = np.array([b]).T
        G = Givens_Reduction(A)
        Q_Tb = np.dot(G[0].T,B)
        return [G[0],G[1],np.linalg.solve(G[1],Q_Tb)]
    #施密特正交化
    elif string == "GS":
        B = np.array([b]).T
        if B.shape[1] != 1:
            return "Input Error!"
        else:
            #m = np.array(A).shape[0]
            n = np.array(A).shape[1]
            X = np.zeros((n,1))

            G = Gram_Schmidt(A)
            Q = G[0]
            R = G[1]
            #np.dot(Q.T,B)
            Y = np.linalg.solve(Q,B)
            X[n-1] = Y[n-1]/R[n-1,n-1]
            for i in range(n-2,-1,-1):
                sum = 0
                for k in range(i+1,n):
                    sum += R[i,k]*X[k]
                X[i] = (Y[i]-sum)/R[i,i]

            return [Q,R,X]




# A = [[2,2,2],[4,7,7],[6,18,22]]
# b= [12,24,12]
# x=Solu(A,b,"LU")
# print(x)
# # k = np.array([u]).T
# Y = np.zeros((3,1))
# print(Y)
#print(np.array([u]).T.shape)


def main():
    if __name__ == "__main__":

        print("请输入矩阵的行数m和矩阵的列数n，用空格键分开m,n:")
        m, n = map(int, input().split())
        print("接下来请输入", m, "*", n, "的矩阵，每个元素用空格分开，以回车键换行")
        A = []
        for i in range(0, m):
            ipt = [int(j) for j in input().split()]
            A.append(ipt)
        print("输入", m, "个值组成b,作为AX的值,每个值用空格分开.")
        arr = input("")
        b = [int(n) for n in arr.split()]
        A_LU = A
        print("行列式|A|计算中:")
        print(determinant(A))