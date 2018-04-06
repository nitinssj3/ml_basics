# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 01:16:16 2018

@author: Nitin
"""

 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 dataset=pd.read_csv('Position_Salaries.csv')
 X=dataset.iloc[: , 1:2].values
 Y=dataset.iloc[: ,2:].values

""""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
"""

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_Y=StandardScaler()
Y = np.ravel(sc_Y.fit_transform(Y.reshape(-1, 1)))

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)


plt.scatter(X,Y,color='blue')
plt.plot(X,regressor.predict(X),color='red')
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))



