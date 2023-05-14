import numpy as np
import copy


class Matrix_Row_Transformation():
    # A是矩阵
    A = [[], []]

    def _init_(self, A):
        self.A = A

    # 第i行和第j行互换
    def Two_Row_Exchange(self, i, j):
        if i == j:
            return self.A
        else:
            temp_row = copy.deepcopy(self.A[i - 1])
            self.A[i - 1] = self.A[j - 1]
            self.A[j - 1] = temp_row

        return self.A

    # 第i行乘以k
    def One_Muti_k(self, i, k):
        self.A[i - 1] = [k * i for i in self.A[i - 1]]

        return self.A

    # 第i行的k倍加到第j行
    def One_Add_Kone(self, i, j, k):
        temp = copy.deepcopy([k * i for i in self.A[i - 1]])
        self.A[j - 1] = list(np.sum([temp, self.A[j - 1]], axis=0))

        return self.A
