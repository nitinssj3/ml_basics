# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

from sklearn.tree import DecisionTreeRegressor 
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

X_grid=np.arange(min(X),max(X),0.001)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='blue')
plt.plot(X_grid,regressor.predict(X_grid),color='red')
