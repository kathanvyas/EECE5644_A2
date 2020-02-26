#!/usr/bin/env python
# coding: utf-8

# In[9]:


#from Create_dataset import create_dataset_2_2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


#setting the size of the plot
plt.rcParams['figure.figsize'] = [5, 5]

################################################

def create_dataset_2_2(flag,prior1,prior2,sample_size,m01,m02,c0_11,c0_12,c0_21,c0_22,m11,m12,c1_11,c1_12,c1_21,c1_22,seed):
    
    np.random.seed(seed)
    
    classes = 2         #total number of classes 
    num = sample_size      #Sample size
    #initializing mean values and setting up mean values
    mean = np.zeros((2,4)) 
    mean[:,0] = [m01,m02] 
    mean[:,1] = [m11,m12]
    
    cov = np.zeros((2,2,4))
    cov[:,:,0] = np.array([[c0_11,c0_12],[c0_21,c0_22]]) 
    cov[:,:,1] = np.array([[c1_11,c1_12],[c1_21,c1_22]])
    
    # priors for two classes
    class_priors = [prior1, prior2]
    
    #creating a class  labels. first we create uniform dirtribution to create labels and then use those 
    #label distribution to select repective guassian distribution to pick data.
    class_labels = np.zeros((1,num))     #initialization
    #uniform distribution creation
    class_labels[0,:] = (np.random.uniform(0,1,num) >= class_priors[0]).astype(int)
    
    X = np.zeros((classes,num))  #initialization of main dataset
    
    # Data Generation Process: Using uniformly distributed Class labels to pick from respective guassian distribution
    for index in range(num):
        if(class_labels[0,index] == 0):
                X[:,index] = np.random.multivariate_normal(mean[:,0],cov[:,:,0],1)
        else:
                X[:,index] = np.random.multivariate_normal(mean[:,1],cov[:,:,1],1) 
                
    if flag == 1:   
        # Code to Plot the actual distribution
        x00 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 0)]
        x01 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 0)]
        x10 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 1)]
        x11 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 1 )]
        plt.plot(X[0,x00],X[1,x00],'.',color ='g')
        plt.plot(X[0,x11],X[1,x11],'+',color = 'r')
        plt.xlabel("Feature x1")
        plt.ylabel("Feature x2")
        plt.legend(["class 0",'class 1'])
        plt.title("Actual Class distribution")
        plt.show()

    
    return X,class_labels,mean,cov #return the transpose of data array (row matrix)

######################################################################################################
######################################################################################################

def min_P_error_classifier(sample_size,class_prior0,class_prior1,dataset,orig_label,gmean,gcov):
    
    #As it is min P(error) classifer, we will always take 0/1 loss
    loss = np.array([[0,1], [1,0]])
    size = sample_size
    prior = [class_prior0,class_prior1]
    
    mean = np.zeros((2,4)) 
    mean[:,0] = gmean[:,0] 
    mean[:,1] = gmean[:,1]
    
    cov = np.zeros((2,2,4))
    cov[:,:,0] = gcov[:,:,0]
    cov[:,:,1] = gcov[:,:,1]
    
    # Gamma/ threshold
    gamma = ((loss[1,0]-loss[0,0])/(loss[1,0] - loss[1,1])) * (prior[0]/prior[1])
    orig_labels = orig_label

    
    new_labels = np.zeros((1,size))
    # Calculation for discriminant score and decisions
    cond_pdf_class0_log = np.log((multivariate_normal.pdf(dataset.T,mean=mean[:,0],cov = cov[:,:,0])))
    cond_pdf_class1_log = np.log((multivariate_normal.pdf(dataset.T,mean=mean[:,1],cov = cov[:,:,1])))
    
    discriminant_score = cond_pdf_class1_log - cond_pdf_class0_log


    new_labels[0,:] = (discriminant_score >= np.log(gamma)).astype(int)

    # Code to plot the distribution after Classification
    x00 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 0 and new_labels[0,i] == 0)]
    x01 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 0 and new_labels[0,i] == 1)]
    x10 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 1 and new_labels[0,i] == 0)]
    x11 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 1 and new_labels[0,i] == 1)]
    plt.plot(dataset[0,x00],dataset[1,x00],'.',color ='g')
    plt.plot(dataset[0,x01],dataset[1,x01],'.',color = 'r')
    plt.plot(dataset[0,x11],dataset[1,x11],'+',color ='g')
    plt.plot(dataset[0,x10],dataset[1,x10],'+',color = 'r')
    plt.legend(["class 0 correctly classified",'class 0 wrongly classified','class 1 correctly classified','class 1 wrongly classified'])
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title('Distribution after classification')
    plt.show()
    
    
    c0 = np.argwhere(orig_labels[0,:]==0).shape[0]
    c1 = np.argwhere(orig_labels[0,:]==1).shape[0]
    #print("Class 0:",c0)
    #print("Class 1:",c1)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    tpr = 0
    fpr = 0
    min_TPR = 0
    min_FPR = 0
    TPR = []
    FPR = []
    new_labels1 = np.zeros((1,size))
    d_labels1 = np.zeros((1,size))
    r=map(lambda x: x/10.0,range(0,500))
    print(r)
    for i in r:
        gamma1 = i
        #print(gamma)
        new_labels1[0,:] = (discriminant_score >= np.log(gamma1)).astype(int)
        #d_labels1[0,:] = discriminant_score >= np.log(gamma)
        for i in range(new_labels1.shape[1]): 
            #print("innerforloop")
            if (orig_labels[0,i] == 1 and new_labels1[0,i] == 1):
               TP += 1
            if (orig_labels[0,i] == 0 and new_labels1[0,i] == 1):
               FP += 1
            if (orig_labels[0,i] == 0 and new_labels1[0,i] == 0):
               TN += 1
            if (orig_labels[0,i] == 1 and new_labels1[0,i] == 0):
               FN += 1
        tpr = TP / (TP+FN)
        fpr = FP / (FP+TN)
        TPR.append(tpr)
        FPR.append(fpr)
        if gamma1 == 9.00000:
            min_TPR = tpr
            min_FPR = fpr
        

    plt.plot(FPR,TPR,'-',color = 'r')
    plt.plot(min_FPR,min_TPR, 'g*')
    plt.legend(["ROC Curve",'Min P Error'])
    plt.show()
    plt.close()
    

    '''
    #h = .01  # step size in the mesh
    # create a mesh to plot in
    hg = np.linspace(np.floor(min(dataset[:,0])),np.ceil(max(dataset[:,0])),1000);
    vg = np.linspace(np.floor(min(dataset[:,1])),np.ceil(max(dataset[:,1])),1000);
    z = np.zeros((1000,1000))
    xy = np.array(np.meshgrid(hg,vg))
    for i in range(100):
        for j in range(100):
            p1 = multivariate_normal.pdf(np.array(xy[0][i][j],xy[1][i][j]),mean=mean[:,1],cov = cov[:,:,1])
            p2 = multivariate_normal.pdf(np.array(xy[0][i][j],xy[1][i][j]),mean=mean[:,0],cov = cov[:,:,0])
            z[i][j] = np.log(p1) - np.log(p2) - np.log(9)
    
    q00 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 0)]
    q01 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 0)]
    q10 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 1)]
    q11 = [i for i in range(class_labels.shape[1]) if (class_labels[0,i] == 1 )]
    plt.plot(dataset[0,q00],dataset[1,q00],'.',color ='g')
    plt.plot(dataset[0,q11],dataset[1,q11],'+',color = 'r')
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.legend(["class 0",'class 1'])
    plt.title("Actual Class distribution")
    plt.show()
    plt.contour(xy[0], xy[1], z)  
    #, cmap=plt.cm.Paired
    '''
#####################################################################
#####################################################################
