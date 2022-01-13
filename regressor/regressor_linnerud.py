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
from scipy.spatial.qhull import _Qhull


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

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#%% hyperparameters
n_samples  = 50
n_neighbor = 14  ## for trimming
test_size  = 0.2
noise      = 0.08 ## original 0.3
n_class    = 3
cl         = 1
#%% prepare 4d data
# =============================================================================
# X = pd.read_csv('datasets/data_banknote_authentication.txt', sep=",", header=None)
# data = np.array(X)
# X_all = data[:,:4]
# y_all = np.reshape((data[:,4]),(-1,1))
# 
# X,X_test,y,y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)
# =============================================================================
print("loading data...")
N = n_samples  ## samples in each class

#(X_ori, y_ori), (X_test_ori, y_test_ori) = mnist.load_data()
# =============================================================================
# mnist = tf.keras.datasets.mnist
# (X_ori, y_ori), (X_test_ori, y_test_ori) = mnist.load_data()
# =============================================================================
from sklearn import datasets

linnerud = datasets.load_linnerud()

#%%
All_train = linnerud.data  # we only take the first two features.
All_test = linnerud.target

All_train = np.array(All_train,dtype=np.float)
All_test = np.array(All_test,dtype=np.float)


X_train = All_train[:,0:2]
y_train = All_train[:,2:]/np.max(All_train[:,2])

X_test = All_test[:,0:2]
y_test = All_test[:,2:]/np.max(All_test[:,2])


pred_all = np.zeros(y_test.shape,y_test.dtype)    

n_features = 2

#%% other machine learning algorithms.
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
    
names = ["KernelRidge", "GaussianProcessRegressor", "KNeighborsRegressor", 
         "DecisionTreeRegressor", "MLPRegressor", "AdaBoostRegressor",
         "SVM"]

classifiers = [
    KernelRidge(alpha=0.1),
    GaussianProcessRegressor(kernel = DotProduct() + WhiteKernel()),
    KNeighborsRegressor(n_neighbors=5),    
    DecisionTreeRegressor(random_state=0),
    MLPRegressor(random_state=1, max_iter=500),
    AdaBoostRegressor(random_state=0, n_estimators=100),
    SVR(C=1.0, epsilon=0.2)
]


for name, clf in zip(names, classifiers):
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    print(name, MSE)
    


#%%
print("1/4 Generate Delaunay Triangularization")

data_1 = X_train

#data_2 = X[d_2]
#print(np.where(y==1))

tri1 = Delaunay(data_1)


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


#%% get predictions

print("3/4 get predictions")
tri_point1 = data_1[tri1_new] #(80, 3, 2)
y_tri = y_train[tri1_new]
#tri_point2 = data_2[tri2_new] #(82, 3, 2)

pred1 = np.zeros_like(y_test)
#pred2 = np.zeros_like(y_test)
## in_hull prediction v2
pred = np.ones(pred_all.shape[0],y_test.dtype)*-1

coordinate_value = np.zeros((X_test.shape[0], tri_point1.shape[0]))

coordinate_all = []

for j in range(tri_point1.shape[0]):
    hulls = tri_point1[j]
    y_hull = y_tri[j]
    #print(j)
#    inhull_detect = in_hull_batch(hulls, X_test)
    
    coordinates = np.dot(np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1), np.linalg.inv(np.concatenate((hulls, np.ones((hulls.shape[0],1))), axis = 1)))
    
    coordinate_all.append(coordinates)
    value = np.sum(np.abs(coordinates), axis=1)
#    value = np.min(np.abs(coordinates), axis=1)
    coordinate_value[:,j] = value
    


rank = np.argsort(coordinate_value, axis=1)

for j in range(rank.shape[0]):
    
    for k in range(29):
        top_close = rank[j,0]
        top_coordinate = coordinate_all[top_close]
        target_bary = top_coordinate[j,:]
        pred[j] = pred[j]+np.dot(target_bary,y_tri[top_close])

print(rank)
    
MSE = mean_squared_error(y_test[:,0], pred)


#print("test boundary time: %.2f second"%(time.time()-time_st2))
print("mse:",MSE)
    

    
