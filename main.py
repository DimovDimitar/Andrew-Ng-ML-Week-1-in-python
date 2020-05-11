# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:20:11 2020

@author: dimit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# insert the data - one X, one y 

data = pd.read_csv('ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
data.head() # view first few rows of the data

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# add the intercept term step

X = X[:,np.newaxis]
y = y[:,np.newaxis]

# initiallise theta matrix of zeroes
theta = np.zeros([2,1])

# parameters for the cost function and gradient descent
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term

def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

# the result of the following will give you circa 37
# for the value of the cost function because it is not optimised
J = computeCost(X, y, theta)
print(J)

def gradientDescentOneVariable(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta

theta = gradientDescent(X, y, theta, alpha, iterations)

print(theta)

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta), 'r')
plt.show()


