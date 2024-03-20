import numpy as np
class linearRegression:
    def fit(self,x,y):
        length_data = len(y)
        x.insert(0,np.ones(length_data))
        X_rawdata = np.array(x)
        X_matrix = X_rawdata.T
        Y_rawdata = np.array([y])
        Y_matrix = Y_rawdata.T
        X_matrix_T = X_matrix.transpose()
        X_T_X = np.matmul(X_matrix_T,X_matrix)
        X_T_X_Inv=np.linalg.inv(X_T_X)
        self.coef = X_T_X_Inv@X_matrix_T@Y_matrix

    def predict(self,x):
        constant = self.coef[0][0]
        y_predict = constant
        coeffs = self.coef[1:]

        if len(coeffs) == len(x):
            for i in range(len(coeffs)):
                y_predict = y_predict + coeffs[i]*x[i]
            print(y_predict)

        else:
            print("the length of input and coeeficients does not match")

lr = linearRegression()
lr.fit(x=[[4,9,8,8,7,5],
    [1,2,3,1,1,0],
    [3,1,2,1,4,2]],y=[52,51,50,52,70,60])

print(lr.coef)
lr.predict([9,2,1])
