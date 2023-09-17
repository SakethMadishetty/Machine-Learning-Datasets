from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data=pd.read_csv('realestatedata.csv')
X=data.iloc[:,[3,4,5,6]]
print(X.head)
y=data.iloc[:,7]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3)

model2=LinearRegression()

model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
print("Library :")
print(r2_score(y_test,prediction2))
