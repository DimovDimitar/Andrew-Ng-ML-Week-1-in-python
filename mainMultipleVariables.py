# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:31:18 2020

@author: dimit
"""


import numpy as np
import pandas as pd

data = pd.read_csv('ex1data2.txt', sep = ',', header = None)
X = data.iloc[:,0:2] # read first two columns into X
y = data.iloc[:,2] # read the third column into y
m = len(y) # no. of training samples
data.head()

# subtract the mean and divide by the standard deviation
X = (X - np.mean(X))/np.std(X) 

# next step is to initialise theta and select the parameters

ones = np.ones((m,1))
X = np.hstack((ones, X))

alpha = 0.01
num_iters = 400

theta = np.zeros((3,1))
y = y[:,np.newaxis]

# perform the first computation for the parameters

def computeCostMulti(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

J = computeCostMulti(X, y, theta)
print(J)

# pretty bad so far because thetha has not been optimised

# use gradient descent for multiple variables to optimise theta

def gradientDescentMulti(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta

theta = gradientDescentMulti(X, y, theta, alpha, num_iters)
print(theta)

# compute J again

J = computeCostMulti(X, y, theta)
print(J)


