# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the data file and import numpy, matplotlib and scipy.

2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

3.Plot the decision boundary.

4.Calculate the y-prediction.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:Arikata Hari Veera Prasad 
RegisterNumber:212223240014
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta ,X ,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iters):
    m=len(y)
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -=alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iters=1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
*/
```

## Output:
### dataset:
![image](https://github.com/Hariveeraprasad-2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145049988/eab111bd-a74b-415c-8e09-66bf30d1eaa4)
### datatypes:
![image](https://github.com/Hariveeraprasad-2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145049988/e3da1238-c326-4072-a357-8b955e7ee014)
### dataset after printing only codes columns:
![image](https://github.com/Hariveeraprasad-2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145049988/98874158-0773-49a8-ace6-6bee43c7df4e)
### Accuracy:
![image](https://github.com/Hariveeraprasad-2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145049988/3879ad89-e2ba-4d20-86ad-7303fabc74f1)
### Array values of Y prediction:
![image](https://github.com/Hariveeraprasad-2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145049988/9eb80633-8b91-43b5-952d-78f5a916934b)
### Array values of Y:
![image](https://github.com/Hariveeraprasad-2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145049988/5fa33274-beb0-4d06-801c-48b4b6040104)
### predicting with different values:
![image](https://github.com/Hariveeraprasad-2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145049988/ea752cbc-2d30-406d-85f7-7ac2923d0efb)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

