#!/usr/bin/env python
# coding: utf-8

# # Question 1 
# ## Creating 4 datastets
# 
# Each of the given datasets are GMMs. The sample size differs. My datastet is created by first creating uniform distribution across all given priors (this were labels) and then picked up points from those labels to select which guassian to pick the data point from. I create a function :
# 1) create_dataset_2_2: this takes all necesssary guassian parametetrs in and gives out a dataset and its labels.
# 
# For finding min P error classifier, I implemented the following mathematical procedure:
# 1) Given: 
#     Data Sample X which has prob P : D_num_train,
#     Conditional Probability x|L: cond_pdf_class0_log,cond_pdf_class1_log,
#     Class Priors: prior,
#     Loss: 0/1 (as min P error)
#     
# 2) Discriminant finction for making decision is based on log likelihood test.
#     

# In[1]:


#from Create_dataset import create_dataset_2_2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
#from fun_lib_c import create_dataset_2_2,min_P_error_classifier,model
from fun_lib import create_dataset_2_2,min_P_error_classifier
from fun_lib2 import model

#setting the size of the plot
plt.rcParams['figure.figsize'] = [5, 5]


# In[8]:


D_10_tr,labels_D_10_tr,m1,c1 = create_dataset_2_2(0,0.9,0.1,10,-2,0,1,-0.9,-0.9,2,2,0,2,0.9,0.9,1,499) #consists 10 samples and their labels for training
D_100_tr,labels_D_100_tr,m2,c2 = create_dataset_2_2(0,0.9,0.1,100,-2,0,1,-0.9,-0.9,2,2,0,2,0.9,0.9,1,499) #consists 100 samples and their labels for training
D_100_tr,labels_D_1000_tr,m3,c3 = create_dataset_2_2(0,0.9,0.1,1000,-2,0,1,-0.9,-0.9,2,2,0,2,0.9,0.9,1,499) #consists 1000 samples and their labels for training
D_10k_V,labels_D_10k_V,m4,c4 = create_dataset_2_2(1,0.9,0.1,10000,-2,0,1,-0.9,-0.9,2,2,0,2,0.9,0.9,1,499) #consists 10K samples and their labels for validation


# In[3]:


#### Q1 Part 1
min_P_error_classifier(10000,0.9,0.1,D_10k_V,labels_D_10k_V,m4,c4)


# ![image.png](attachment:image.png)

# In[10]:


######### Q1 Part 2 
######## D_10_train
p = np.array([[0.9,0.1]]);
m = 10
np.random.seed(404)
y = np.where(np.reshape(np.random.random((1,m)), (m,1))>=p[0][0],1,0)
num0 = np.argwhere(y==0).shape[0];
num1 = np.argwhere(y==1).shape[0];
m0 = np.array([-2,0]);
c0 = np.array([[1,0],[0,3]]); 
n0=num0;
x0 = np.random.multivariate_normal(m0,c0,n0);
m1 = np.array([2,0]);
c1 = np.array([[1,0],[0,3]]);n1=num1;
x1 = np.random.multivariate_normal(m1,c1,n1);
test_set_x = train_set_x = np.concatenate((x0, x1), axis=0).T;
test_set_y = train_set_y = np.concatenate((np.zeros((5, 1)),np.ones((5,1))), axis=0).T;
print("dataset:", test_set_x.shape);
print("Labels_set:",test_set_y.shape);
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# In[11]:


######### Q1 Part 2:
#### D_100_train
p = np.array([[0.9,0.1]]);
m = 100
np.random.seed(404)
y = np.where(np.reshape(np.random.random((1,m)), (m,1))>=p[0][0],1,0)
num0 = np.argwhere(y==0).shape[0];
num1 = np.argwhere(y==1).shape[0];
m0 = np.array([-2,0]);
c0 = np.array([[1,0],[0,3]]); 
n0=num0;
x0 = np.random.multivariate_normal(m0,c0,n0);
m1 = np.array([2,0]);
c1 = np.array([[1,0],[0,3]]);n1=num1;
x1 = np.random.multivariate_normal(m1,c1,n1);
test_set_x = train_set_x = np.concatenate((x0, x1), axis=0).T;
test_set_y = train_set_y = np.concatenate((np.zeros((50, 1)),np.ones((50,1))), axis=0).T;
print("dataset:", test_set_x.shape);
print("Labels_set:",test_set_y.shape);
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.0005, print_cost = True)


# In[12]:


######### Q1 Part 2
###### D_1000_train
p = np.array([[0.9,0.1]]);
m = 1000
np.random.seed(404)
y = np.where(np.reshape(np.random.random((1,m)), (m,1))>=p[0][0],1,0)
num0 = np.argwhere(y==0).shape[0];
num1 = np.argwhere(y==1).shape[0];
m0 = np.array([-2,0]);
c0 = np.array([[1,0],[0,3]]); 
n0=num0;
x0 = np.random.multivariate_normal(m0,c0,n0);
m1 = np.array([2,0]);
c1 = np.array([[1,0],[0,3]]);n1=num1;
x1 = np.random.multivariate_normal(m1,c1,n1);
test_set_x = train_set_x = np.concatenate((x0, x1), axis=0).T;
test_set_y = train_set_y = np.concatenate((np.zeros((500, 1)),np.ones((500,1))), axis=0).T;
print("dataset:", test_set_x.shape);
print("Labels_set:",test_set_y.shape);
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.00000000005, print_cost = True)


# In[13]:


######### Q1 Part 2:
######### Validating on 10K dataset

p = np.array([[0.9,0.1]]);
m = 10000
np.random.seed(404)
y = np.where(np.reshape(np.random.random((1,m)), (m,1))>=p[0][0],1,0)
num0 = np.argwhere(y==0).shape[0];
num1 = np.argwhere(y==1).shape[0];
m0 = np.array([-2,0]);
c0 = np.array([[1,0],[0,3]]); 
n0=num0;
x0 = np.random.multivariate_normal(m0,c0,n0);
m1 = np.array([2,0]);
c1 = np.array([[1,0],[0,3]]);n1=num1;
x1 = np.random.multivariate_normal(m1,c1,n1);
test_set_x = train_set_x = np.concatenate((x0, x1), axis=0).T;
test_set_y = train_set_y = np.concatenate((np.zeros((5000, 1)),np.ones((5000,1))), axis=0).T;
print("dataset:", test_set_x.shape);
print("Labels_set:",test_set_y.shape);
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.00000000005, print_cost = True)


# In[ ]:




