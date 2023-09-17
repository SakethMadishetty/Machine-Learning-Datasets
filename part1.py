import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sb
from sklearn.metrics import r2_score
root = tk.Tk()
root.withdraw()

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        df=pd.DataFrame(columns=["i","MSE"])
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for iteration in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            current_error = np.mean((y_pred - y) ** 2)
            row={'i': iteration, 'MSE': current_error}
            df=pd.concat([df,pd.DataFrame([row])])

        # print(df.to_string())
        plt.plot(df['i'],df['MSE'])
        plt.xlabel("Iteration")
        plt.ylabel("Mean Squared Error")
        plt.title("No. of iterations (vs) Mean Squared Error")
        plt.show()

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def mean_squared_error(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse


def load_datasets():
    github_raw_url = "https://raw.githubusercontent.com/SakethMadishetty/Machine-Learning-Datasets/main/realestatedata.csv"
    df = pd.read_csv(github_raw_url)
    return df

def normalise(df, column_index):
    column_to_normalize = df.iloc[:, column_index]
    min_value = column_to_normalize.min()
    max_value = column_to_normalize.max()
    df.iloc[:, column_index] = (column_to_normalize - min_value) / (max_value - min_value)
    return df

def sampling(X, y, test_size=0.3, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    num_samples = X.shape[0]
    num_test_samples = int(test_size * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_indices = indices[:num_test_samples]
    test_indices = indices[num_test_samples:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

###############################################################################################################################

data=load_datasets()

data=data.dropna()  #removing null values
data = data.drop_duplicates()  #removing duplicate rows
# print(data.head())  #visualizing the data frame, if there are any categorial variables they have to be encoded
# no categorical variables
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
fig, axes = plt.subplots(2, 3, figsize=(screen_width/100,screen_height/100)) # 2rows 3columns
plt.title("Scatter plots depicts how every Independent Variable is related to Dependent Variable")

axes[0,0].scatter(data.iloc[:,1],data.iloc[:,7],color='red',alpha=0.5,marker=".")
axes[0,0].set_xlabel("Transaction date")
axes[0,0].set_ylabel("House price of unit area")
axes[0,0].set_title('Transaction date (vs) house price per unit area',fontsize=14)


axes[0,1].scatter(data.iloc[:,2],data.iloc[:,7],color='green',alpha=0.5,marker="v")
axes[0,1].set_xlabel("House Age")
axes[0,1].set_ylabel("House price of unit area")
axes[0,1].set_title('House Age (vs) house price per unit area',fontsize=14)

axes[0,2].scatter(data.iloc[:,3],data.iloc[:,7],color='orange',alpha=0.8,marker="^")
axes[0,2].set_xlabel("Distance to the nearest MRT station")
axes[0,2].set_ylabel("House price of unit area")
axes[0,2].set_title('Distance to the nearest MRT station (vs) house price per unit area',fontsize=14)

axes[1,0].scatter(data.iloc[:,4],data.iloc[:,7],color='violet',alpha=0.8,marker="+")
axes[1,0].set_xlabel("Number of convenience stores")
axes[1,0].set_ylabel("House price of unit area")
axes[1,0].set_title('Number of convenience stores (vs) house price per unit area',fontsize=14)

axes[1,1].scatter(data.iloc[:,5],data.iloc[:,7],color='blue',alpha=0.8,marker="p")
axes[1,1].set_xlabel("Latitude")
axes[1,1].set_ylabel("House price of unit area")
axes[1,1].set_title('Latitude (vs) house price per unit area',fontsize=14)

axes[1,2].scatter(data.iloc[:,6],data.iloc[:,7],color='maroon',alpha=0.6,marker="*")
axes[1,2].set_xlabel("Longitude")
axes[1,2].set_ylabel("House price of unit area")
axes[1,2].set_title('Longitude (vs) house price per unit area',fontsize=14)
plt.subplots_adjust(wspace=0.5,hspace=0.4)
plt.show()
plt.close()
#data inspection
print(data.describe().to_string())

# finding correlation between independent variales and dependent variable
plt.figure(figsize=(screen_width/100,screen_height/100))
corr=data.corr()
sb.heatmap(corr,cmap='coolwarm',annot=True)
sb.set (rc = {'figure.figsize':(9, 8)})
plt.show()
plt.close()
print("\n\n",corr.iloc[:,7].to_string())
print("Transaction date and house age are less correlated with house price per unit area")

# independent variable
X=data.iloc[:,[3,4,5,6]]
print(X.to_string())
X=normalise(X,0)
X=normalise(X,1)
X=normalise(X,2)
X=normalise(X,3)
print(X.to_string())

#Dependent variable
y=data.iloc[:,7]
print(y.to_string())

X_train,X_test,y_train,y_test= sampling(X,y,test_size=0.1)

model = LinearRegression(learning_rate=0.05, n_iterations=1000)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(prediction)

print("\n\n\nCoefficients (weights):", model.weights)
print("Intercept (bias):", model.bias)
print("\n\n")
Train_mse=model.mean_squared_error(y_train,model.predict(X_train))
Train_r2=r2_score(y_train,model.predict(X_train))
print("The meansquared error on Train data: ",Train_mse)
print("The r squared error on Train data: ",Train_r2)

print("\n\n")
print("The meansquared error on test data: ",model.mean_squared_error(y_test,prediction))
print("The r squared error on test data: ",r2_score(y_test,prediction))



##################################################################################
