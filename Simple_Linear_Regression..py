# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 01:33:18 2018

@author: Nitin
"""

 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 dataset=pd.read_csv('Salary_Data.csv')
 X=dataset.iloc[: , :-1].values
 Y=dataset.iloc[: , 1].values


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit(X_test)"""
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)

plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('salary vs exp')
plt.xlabel('year of experience')
plt.ylabel('salary')
 plt.show()
 
 plt.scatter(X_test,Y_test,color='blue')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('salary vs exp')
plt.xlabel('year of experience')
plt.ylabel('salary')
 plt.show()