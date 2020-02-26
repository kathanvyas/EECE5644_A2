#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from Create_dataset import create_dataset_2_2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


#setting the size of the plot
plt.rcParams['figure.figsize'] = [5, 5]

def h_fun(z):
    s = 1/(1+np.exp(-z))    
    return s

def param_initial(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b
    
lam0 = np.ones((2,1));

###############################

def propagate(w, b, X, Y):
    lam0 = np.ones((2,1));
    m = X.T.shape[1];
    # FORWARD PROPAGATION (FROM X TO COST)
    A = h_fun(np.dot(w.T, X)+b); # compute activation
    cost = -1/m*sum(np.squeeze(lam0[0,0]*Y*np.log(A)+lam0[1,0]*(1-Y)*np.log(1-A)))   # compute cost
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m*(np.dot(X,(A-Y).T))
    db = 1/m*sum(np.squeeze(A-Y))
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost

###############################

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations): 
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule
        w = w-learning_rate*dw
        b = b-learning_rate*db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,"b": b}
    grads = {"dw": dw,"db": db}
    return params, grads, costs

##############################

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = h_fun(np.dot(w.T,X)+b)  
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[:,i] <= 0.5:
            A[:,i]=0
        else:
            A[:,i]=1
    Y_prediction=A
    assert(Y_prediction.shape == (1, m))   
    return Y_prediction
    
#####################################################################
#####################################################################

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # initialize parameters with zeros
    w, b = param_initial(X_train.shape[0])
    # Gradient descent
    #print(w.shape[0])
    #print(b.shape[0])
    #print(w.shape)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d


# In[ ]:




