# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:40:34 2021

@author: Dayang_Wang
"""
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

#from mayavi import mlab

import time
from utils import *

#%% for comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import random
random.seed(10)

#%% hyperparameters
n_samples  = 50
n_neighbor = 14  ## for trimming
test_size  = 0.2
noise      = 0.08 ## original 0.3
n_class    = 2

#%% prepare 4d data

print("loading data...")



XA = np.loadtxt("Train_RingSpiral_A.txt",  delimiter=' ')

XB = np.loadtxt("Train_RingSpiral_B.txt",  delimiter=' ')


yA = np.zeros((XA.shape[0],))

yB = np.ones((XB.shape[0],))

XA1 = np.loadtxt("Train_RingSpiral_1_A.txt",  delimiter=' ')

XB1 = np.loadtxt("Train_RingSpiral_1_B.txt",  delimiter=' ')

yA1 = np.zeros((XA1.shape[0],))

yB1 = np.ones((XB1.shape[0],))



XA = np.concatenate((XA, XA1), axis = 0)
XB = np.concatenate((XB, XB1), axis = 0)

yA = np.concatenate((yA, yA1), axis=0)

yB = np.concatenate((yB, yB1), axis=0)


X = np.concatenate((XA, XB), axis = 0)
y = np.concatenate((yA, yB), axis = 0)

X = X + np.random.randn(X.shape[0],3)*0.05

XtestA = np.loadtxt("Test_RingSpiral_A.txt",  delimiter=' ')

XtestB = np.loadtxt("Test_RingSpiral_B.txt",  delimiter=' ')

ytestA = np.zeros((XtestA.shape[0],))

ytestB = np.ones((XtestB.shape[0],))



X_test = np.concatenate((XtestA, XtestB), axis = 0)

y_test = np.concatenate((ytestA, ytestB), axis = 0)



data_all = []
dictionary_all = []


print(X.shape)

pred_all = np.zeros((y_test.shape[0],n_class), y_test.dtype)    
#%% other machine learning algorithms.
y_c = y.reshape(X.shape[0])  ## change the shape of y to 1d n_samples
y_test_c = y_test.reshape(X_test.shape[0])
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(X, y_c)
    score = clf.score(X_test, y_test_c)
    print(name,score)

#%% start training. 
for cl in range(n_class):
    start_time = time.time()
    print("Dealing with class %d/%d" %(cl,n_class-1))
    #%% 1/4 Generate Delaunay Triangularization
    print("1/4 Generate Delaunay Triangularization")
    d_1 = np.where(y==cl) ## the id of the points, 0,1,2,3... . d_1,_ = np.where(y==cl)
    #d_2,_ = np.where(y==0)

    data_1 = X[d_1]
    data_all.append(data_1)
    #data_2 = X[d_2]
    #print(np.where(y==1))
    
    tri1 = Delaunay(data_1)
    #tri2 = Delaunay(data_2)
    
    #print(tri1.simplices.shape)
    #print(tri2.simplices.shape)
    
    #%% Manifold Trimming
    print("2/4 Manifold Trimming")
    neighbors1, distances = distance_mat(data_1,n_neighbor) ## 14 original 
    #neighbors2, distances = distance_mat(data_2,n_neighbor)
    
    TriangleForTrimming1 = tri1.simplices

    #TriangleForTrimming2 = tri2.simplices
    #print(TriangleForTrimming1.shape)
    num1,dim1 = TriangleForTrimming1.shape
    tri1_new = []
    for i in np.arange(num1):
        triangle = TriangleForTrimming1[i,:]
        checker = 1
        for j in range(dim1-1):
            for k in range(j+1,dim1):
        # any of the line(two points) in a triangle must has a connection(>0)
                if neighbors1[triangle[j],triangle[k]]==0:
                #if (neighbors1[triangle[0],triangle[1]]>0) & (neighbors1[triangle[2],triangle[1]]>0) & (neighbors1[triangle[2],triangle[0]]>0):
                    checker = 0
                    break;
        if checker == 1:
            tri1_new.append(triangle)


    tri1_new = np.array(tri1_new)    
    #print(tri1_new.shape)
    #print(tri2_new.shape)


# =============================================================================
#     mlab.triangular_mesh(data_1[:,0], data_1[:,1], data_1[:,2], tri1_new)
# 
#     #for i in np.arange(int(n_samples*(1-test_size)//2)):
#       #plt.annotate(i, (data_2[i,0], data_2[i,1])) # annotate the given points
#     mlab.show()
# 
# =============================================================================
    
    #%% Detecting the Envelop of the Triangularization
    print("3/4 Detecting the Envelop")
    
    #dictionary1 = np.zeros((data_1.shape[0],data_1.shape[0])) # original amount to be zero
    dictionary1 = [] ## use the diction to record all combinations of hyperplane  (n,4)
    #print(dictionary1.shape)
    tr1,con1 = tri1_new.shape
    for j in np.arange(tr1):
        triangle = tri1_new[j,:]
        
        for i in range(con1):
            hyperplane1 = np.array([triangle[(i+1)%con1],triangle[(i+2)%con1],triangle[(i+3)%con1],triangle[(i+4)%con1]]) ## three dimension
            dictionary1.append(hyperplane1)  
    dictionary1 = np.array(dictionary1)
    dictionary_all.append(dictionary1)    
    ## sort the record and keep unique combinations:
    #ind1 = np.argsort(dictionary1, axis=1)
    #dictionary1 = np.take_along_axis(dictionary1, ind1, axis=1)
    dictionary1 = np.sort(dictionary1,axis=1)
    #ind2 = np.argsort(dictionary2, axis = 1)
    #dictionary2 = np.take_along_axis(dictionary2, ind2, axis=1)
    
    arr, uniq_cnt = np.unique(dictionary1, axis=0, return_counts=True)
    dictionary1 = arr[uniq_cnt==1]
    #arr, uniq_cnt = np.unique(dictionary2, axis=0, return_counts=True)
    #dictionary2 = arr[uniq_cnt==1]
    #print(dictionary1.shape)
    #print(dictionary2.shape)
    #%% get predictions
    print("4/4 get predictions")
    tri_point1 = data_1[tri1_new] #(80, 3, 2)
    #tri_point2 = data_2[tri2_new] #(82, 3, 2)
    
    pred1 = np.zeros_like(y_test)
    #pred2 = np.zeros_like(y_test)


    for j in range(tri_point1.shape[0]):
            hulls = tri_point1[j]
            inhull_detect = in_hull_batch(hulls, X_test)
            pred_all[:,cl] = 1 * inhull_detect
            

    print("execute time: %.2f seconds"%(time.time()-start_time))

dist_max = np.zeros(pred_all.shape)  ## need the max
dist_min = np.ones(pred_all.shape)*99999  ## need the min
pred = np.ones(pred_all.shape[0],y_test.dtype)*-1

whole_test = y_test.shape[0]
outhull = 0
time_st2 = time.time()
#print(pred_all.shape)
for i in range(pred_all.shape[0]):
    #print("sum",sum(pred_all[i]))
    if sum(pred_all[i]) == 0: ## if in no simplex
        for ind,(data_1,dictionary1) in enumerate(zip(data_all,dictionary_all)):
            dist_min[i,ind] = point_to_envelop_fast2(data_1,dictionary1,X_test[i])
        pred[i] = np.amin(np.where(dist_min[i] == np.amin(dist_min[i])))
        outhull = outhull + 1
    elif sum(pred_all[i]) == 1:
        #print(np.where(pred_all[i] == 1))
        pred[i] = np.amin(np.where(pred_all[i] == 1))
    else:
        #for ind in range(n_class):
        for ind,(data_1,dictionary1) in enumerate(zip(data_all,dictionary_all)):
            if pred_all[i,ind] == 1:
                dist_max[i,ind] = point_to_envelop_fast2(data_1,dictionary1,X_test[i])
        pred[i] = np.amin(np.where(dist_max[i] == np.amax(dist_max[i])))
    #print(pred[i])
#show_res = np.concatenate((y_test.reshape((-1,1)),pred.reshape((-1,1))),axis = 1)
correct = np.mean(pred.reshape((-1,1)) == y_test.reshape((-1,1)))
in_rate = 1 - np.float32(outhull)/whole_test

print("test boundary time: %.2f second"%(time.time()-time_st2))
print("in hull rate:", in_rate)
print("acc:",correct)
   

