# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:36:51 2020

@author: dimit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # more on this later

data = pd.read_csv('ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
data.head()
m = len(y)

# visualise

mask = y == 1
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

# add the sigmoid function for logistic reg

def sigmoid(x):
  return 1/(1+np.exp(-x))

# cost function including the sigmoid func

def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) 
        + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J

# gradient function

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))

# calling functions using initial parameters
    
(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = costFunction(theta, X, y)
print(J)

# using scipy's built in function fmin_tnc to optimise for theta

temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))

# the output of above function is a tuple whose first element
#contains the optimized values of theta

theta_optimized = temp[0]
print(theta_optimized)

#    Note on flatten() function: 
# Unfortunately scipy’s fmin_tnc doesn’t work well with column 
# or row vector. It expects the parameters to be in an array 
# format. The flatten() function reduces a column or row vector 
# into array format.

J = costFunction(theta_optimized[:,np.newaxis], X, y)
print(J)

# plot decision boundary

plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] + np.dot(theta_optimized[1],plot_x)) 

mask = y.flatten() == 1
adm = plt.scatter(X[mask][:,1], X[mask][:,2])
not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2])
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)
    
accuracy(X, y.flatten(), theta_optimized, 0.5)

