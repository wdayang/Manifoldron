# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:35:56 2021

@author: Dayang_Wang
"""

import numpy as np
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from numpy import pi

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

#%% hyperparameters
n_samples  = 100
n_neighbor = 14  ## for trimming
test_size  = 0.4
noise      = 0.3 ## original 0.3

#%% setup dataset
#datasets = make_moons(n_samples=n_samples, noise=noise, random_state=0)
#datasets = make_circles(n_samples=n_samples,noise=noise, random_state=0,factor=0.5)
#datasets = make_classification(random_state=0)
# =============================================================================
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# datasets = (X, y)
# =============================================================================
N = 400
theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)

r_a = 2*theta + pi
data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x_a = data_a + np.random.randn(N,2)

r_b = -2*theta - pi
data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
x_b = data_b + np.random.randn(N,2)*1.5

res_a = np.append(x_a, np.zeros((N,1)), axis=1)
res_b = np.append(x_b, np.ones((N,1)), axis=1)

res = np.append(res_a, res_b, axis=0)
np.random.shuffle(res)

X_all = res[:,:2]
y_all = np.reshape((res[:,2]),(-1,1))

X,X_test,y,y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

#%% other machine learning algorithms.
y_c = y.reshape(X.shape[0])  ## change the shape of y to 1d n_samples
y_test_c = y_test.reshape(X_test.shape[0])
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
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

#%% Generate Delaunay Triangularization
d_1,_ = np.where(y==1) ## there is no two x to be the same, so this is ok
d_2,_ = np.where(y==0)

data_1 = X[d_1,:]
data_2 = X[d_2,:]

tri1 = Delaunay(data_1)
tri2 = Delaunay(data_2)

plt.figure()
plt.triplot(data_1[:,0], data_1[:,1], tri1.simplices) #Draw a unstructured triangular grid as lines and/or markers.
plt.plot(data_1[:,0], data_1[:,1], 'ro')
plt.triplot(data_2[:,0], data_2[:,1], tri2.simplices)
plt.plot(data_2[:,0], data_2[:,1], 'bo')
plt.show()

#%% for test:
d_11,_ = np.where(y_test==1) ## there is no two x to be the same, so this is ok
d_22,_ = np.where(y_test==0)
X_test1 = X_test[d_11,:]
X_test2 = X_test[d_22,:]

#%% Manifold Trimming
neighbors1, distances = distance_mat(data_1,n_neighbor) ## 14 original 
neighbors2, distances = distance_mat(data_2,n_neighbor)

TriangleForTrimming1 = tri1.simplices
TriangleForTrimming2 = tri2.simplices

tri1_new = []
for j in np.arange(TriangleForTrimming1.shape[0]):
    triangle = TriangleForTrimming1[j,:]
    # any of the line(two points) in a triangle must has a connection(>0)
    if (neighbors1[triangle[0],triangle[1]]>0) & (neighbors1[triangle[2],triangle[1]]>0) & (neighbors1[triangle[2],triangle[0]]>0):
        tri1_new.append(triangle)
tri1_new = np.array(tri1_new)

tri2_new = []
for j in np.arange(TriangleForTrimming2.shape[0]):
    triangle = TriangleForTrimming2[j,:]
    if (neighbors2[triangle[0],triangle[1]]>0) & (neighbors2[triangle[2],triangle[1]]>0) & (neighbors2[triangle[2],triangle[0]]>0):
        tri2_new.append(triangle)
tri2_new = np.array(tri2_new)

plt.figure()
plt.triplot(data_1[:,0], data_1[:,1], tri1_new)
plt.plot(data_1[:,0], data_1[:,1], 'ro')
#for i in np.arange(int(n_samples*(1-test_size)//2)):
  #print(i)
  #plt.annotate(i, (data_1[i,0], data_1[i,1]))
    
plt.triplot(data_2[:,0], data_2[:,1], tri2_new)
plt.plot(data_2[:,0], data_2[:,1], 'bo')
#for i in np.arange(int(n_samples*(1-test_size)//2)):
  #plt.annotate(i, (data_2[i,0], data_2[i,1])) # annotate the given points
plt.show()

#%% Detecting the Envelop of the Triangularization

dictionary1 = np.zeros((data_1.shape[0],data_1.shape[0])) # original amount to be zero
#print(dictionary1.shape)
for j in np.arange(tri1_new.shape[0]):
    triangle = tri1_new[j,:]
         
    dictionary1[(triangle[0],triangle[1])] += 1
    dictionary1[(triangle[0],triangle[2])] += 1
    dictionary1[(triangle[2],triangle[1])] += 1
    
    dictionary1[(triangle[1],triangle[0])] += 1
    dictionary1[(triangle[2],triangle[0])] += 1
    dictionary1[(triangle[1],triangle[2])] += 1       


dictionary2 = np.zeros((data_2.shape[0],data_2.shape[0]))
for j in np.arange(tri2_new.shape[0]):
    triangle = tri2_new[j,:]
         
    dictionary2[(triangle[0],triangle[1])] += 1
    dictionary2[(triangle[0],triangle[2])] += 1
    dictionary2[(triangle[2],triangle[1])] += 1    

    dictionary2[(triangle[1],triangle[0])] += 1
    dictionary2[(triangle[2],triangle[0])] += 1
    dictionary2[(triangle[1],triangle[2])] += 1   

## there are many duplicates
aa1, bb1 = np.where(dictionary1==1) # get the point index where the line is only in one triangle
aa2, bb2 = np.where(dictionary2==1)

plt.figure()
for k in np.arange(len(bb1)): # two consequent point here and the line, why -2
    #plt.plot([data_1[aa1[k:k+1],0],data_1[bb1[k:k+1],0]],[data_1[aa1[k:k+1],1],data_1[bb1[k:k+1],1]], 'r-o')
    plt.plot([data_1[aa1[k],0],data_1[bb1[k],0]], [data_1[aa1[k],1],data_1[bb1[k],1]], 'r-o')

for k in np.arange(len(bb2)):
    #plt.plot([data_2[aa2[k:k+1],0],data_2[bb2[k:k+1],0]], [data_2[aa2[k:k+1],1],data_2[bb2[k:k+1],1]], 'b-o')
    plt.plot([data_2[aa2[k],0],data_2[bb2[k],0]], [data_2[aa2[k],1],data_2[bb2[k],1]], 'b-o')


envelop1_u, indices = np.unique(np.array(aa1), return_inverse=True) ## get rid of the duplicates
envelop2_u, indices = np.unique(np.array(aa2), return_inverse=True)
    
plt.plot(data_1[envelop1_u,0], data_1[envelop1_u,1], 'ro')
plt.plot(data_2[envelop2_u,0], data_2[envelop2_u,1], 'bo')

plt.plot(X_test1[:,0], X_test1[:,1], 'ro')
plt.plot(X_test2[:,0], X_test2[:,1], 'bo')
plt.show()
#%% get predictions
tri_point1 = data_1[tri1_new] #(80, 3, 2)
tri_point2 = data_2[tri2_new] #(82, 3, 2)

pred1 = np.zeros_like(y_test)
pred2 = np.zeros_like(y_test)

for i in range(X_test.shape[0]):
  x0,y0 = X_test[i,0],X_test[i,1]
  for j in range(tri_point1.shape[0]):
    [[x1,y1],[x2,y2],[x3,y3]] = tri_point1[j]
    if PointDeter(x0,y0,x1,y1,x2,y2,x3,y3):
      pred1[i] = 1
      break

  for j in range(tri_point2.shape[0]):
    [[x1,y1],[x2,y2],[x3,y3]] = tri_point2[j]
    if PointDeter(x0,y0,x1,y1,x2,y2,x3,y3):
      pred2[i] = 1
      break

pred = np.zeros_like(y_test)
whole_test = y_test.shape[0]
inhull = 0

for i in range(y_test.shape[0]):
  if pred1[i] == 1 and pred2[i] == 0:
    pred[i] = 1
    inhull = inhull + 1
  elif pred1[i] == 0 and pred2[i] == 1:
    pred[i] = 0
    inhull = inhull + 1
  elif pred1[i] == 0 and pred2[i] == 0:
    pred[i] = CloseLine(data_1,data_2,aa1,bb1,aa2,bb2,X_test[i,0],X_test[i,1])
  elif pred1[i] == 1 and pred2[i] == 1: # if is included in both manifolds, get the fall into the most far partition.
    pred[i] = 1 - CloseLine(data_1,data_2,aa1,bb1,aa2,bb2,X_test[i,0],X_test[i,1])

correct = np.mean(pred == y_test)
#%% Verify predictions
d_11,_ = np.where(pred==1) ## there is no two x to be the same, so this is ok
d_22,_ = np.where(pred==0)
X_test11 = X_test[d_11,:]
X_test22 = X_test[d_22,:]

plt.figure()
for k in np.arange(len(bb1)): # two consequent point here and the line, why -2
    #plt.plot([data_1[aa1[k:k+1],0],data_1[bb1[k:k+1],0]],[data_1[aa1[k:k+1],1],data_1[bb1[k:k+1],1]], 'r-o')
    plt.plot([data_1[aa1[k],0],data_1[bb1[k],0]], [data_1[aa1[k],1],data_1[bb1[k],1]], 'r-o')
for k in np.arange(len(bb2)):
    #plt.plot([data_2[aa2[k:k+1],0],data_2[bb2[k:k+1],0]], [data_2[aa2[k:k+1],1],data_2[bb2[k:k+1],1]], 'b-o')
    plt.plot([data_2[aa2[k],0],data_2[bb2[k],0]], [data_2[aa2[k],1],data_2[bb2[k],1]], 'b-o')

envelop1_u, indices = np.unique(np.array(aa1), return_inverse=True)
envelop2_u, indices = np.unique(np.array(aa2), return_inverse=True)
    
plt.plot(data_1[envelop1_u,0], data_1[envelop1_u,1], 'ro')
plt.plot(data_2[envelop2_u,0], data_2[envelop2_u,1], 'bo')

plt.plot(X_test11[:,0], X_test11[:,1], 'ro')
plt.plot(X_test22[:,0], X_test22[:,1], 'bo')
plt.show()
#%% print acc and wrong predictions

#print(np.where(pred!=y_test))
print('prediction acc:',correct)
in_rate = float(inhull)/whole_test
print("in hull rate:", in_rate)
print("wrong prediction point:")
print(X_test[np.where(pred!=y_test),:])