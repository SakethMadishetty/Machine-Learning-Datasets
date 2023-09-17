from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def load_datasets():
    github_raw_url = "https://raw.githubusercontent.com/SakethMadishetty/Machine-Learning-Datasets/main/realestatedata.csv"
    df = pd.read_csv(github_raw_url)
    return df

data=load_datasets()

X=data.iloc[:,[3,4,5,6]]
# print(X.head)
y=data.iloc[:,7]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3)

model2=LinearRegression()

model2.fit(X_train, y_train)

y_pred_train = model2.predict(X_train)

print("Coefficients/weights : ",model2.coef_)
print("intercept : ",model2.intercept_)

mse_on_train = mean_squared_error(y_train, y_pred_train)
print("Mean squared error for predictions on training data: ", mse_on_train)
print("R2 score on predictions on training data:",r2_score(y_train, y_pred_train))

prediction2 = model2.predict(X_test)
print("Mean squared error for predictions on testing data: ", mean_squared_error(y_test,prediction2))
print("Coefficient of determination / R2 Score :")
print(r2_score(y_test,prediction2))