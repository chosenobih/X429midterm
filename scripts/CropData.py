'''
Name: Nicholas Stout
ISTA 429 midterm
Date: October 12, 2021
Collaboration:
'''
import numpy as np
import pandas as pd

data = np.load('inputs_others_train.npy')
X_train = pd.DataFrame(data)

print(X_train.head())

data2 = np.load('yield_train.npy')
Y_train = pd.DataFrame(data2)

print(Y_train.head())

X_train[0] = pd.to_numeric(X_train[0])
X_train[1] = pd.to_numeric(X_train[1])
X_train[3] = pd.to_numeric(X_train[3])
X_train[4] = pd.to_numeric(X_train[4])

stateValues = {}
counter = 0
for y in X_train[2]:
    if (y in stateValues.keys()):
        continue
    else:
        stateValues[y] = counter
        counter += 1
X_train[2].replace(stateValues, inplace = True)
    

print(X_train.info())
print(X_train.head())

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

trainingSetX, testingSetX, trainingSetY, testingSetY = train_test_split(X_train, Y_train, test_size = 0.3, random_state = 101)

model = LinearRegression()
model.fit(trainingSetX, trainingSetY)

predictions = []
predictions = model.predict(trainingSetX)
print(np.sqrt(sklearn.metrics.mean_squared_error(trainingSetY, predictions)))

predictions = []
predictions = model.predict(testingSetX)
print(np.sqrt(sklearn.metrics.mean_squared_error(testingSetY, predictions)))

from sklearn.linear_model import BayesianRidge
model2 = BayesianRidge()
model2.fit(trainingSetX, trainingSetY)

predictions = []
predictions = model.predict(testingSetX)
print(np.sqrt(sklearn.metrics.mean_squared_error(testingSetY, predictions)))

from sklearn.neighbors import KNeighborsRegressor
model3 =  KNeighborsRegressor(n_neighbors = 5, algorithm= 'brute')
model3.fit(trainingSetX, trainingSetY)

predictions = []
predictions = model.predict(testingSetX)
print(np.sqrt(sklearn.metrics.mean_squared_error(testingSetY, predictions)))