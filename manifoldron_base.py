import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.model_selection import train_test_split

import time
from utils import *

#%% for comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#%% hyperparameters
n_samples  = 14
test_size  = 0.3

#%% prepare 4d data
print("loading data...")
data = pd.read_csv('data/tic-tac-toe.csv')
data.head()
X_all = np.array(data[["TL","TM","TR","ML", "MM","MR", "BL","BM","BR"]])
y_all = np.array(data["class"])
X_all[X_all=='x'] = 1
X_all[X_all=='o'] = 2
X_all[X_all=='b'] = 3

X_all = X_all + np.random.normal(0,0.1,X_all.shape)
y_all[y_all==True] = 1
y_all[y_all==False] = 0

X_all = np.array(X_all,dtype=np.float)
y_all = np.array(y_all,dtype=np.float)

#%%
X,X_test,y,y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

n_features = X.shape[1]
n_class = np.unique(y).shape[0]

data_all = []
dictionary_all = []
print(X.shape)

pred_all = np.zeros((y_test.shape[0],n_class),y_test.dtype)    
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

#%% start training. 
for cl in range(n_class):
    start_time = time.time()
    print("Dealing with class %d/%d" %(cl,n_class-1))
    #%% 1/4 Generate Delaunay Triangularization
    print("1/4 Generate Delaunay Triangularization")
    d_1 = np.where(y==cl) ## the id of the points, 0,1,2,3... . d_1,_ = np.where(y==cl)

    data_1 = X[d_1]
    data_all.append(data_1)
    tri1 = Delaunay(data_1)
    
    #%% Manifold Trimming
    print("2/4 Manifold Trimming")
    neighbors1, distances = distance_mat(data_1,n_neighbor) ## 14 original 
    TriangleForTrimming1 = tri1.simplices
    num1,dim1 = TriangleForTrimming1.shape
    tri1_new = []
    for i in np.arange(num1):
        triangle = TriangleForTrimming1[i,:]
        checker = 1
        for j in range(dim1-1):
            for k in range(j+1,dim1):
                if neighbors1[triangle[j],triangle[k]]==0:
                    checker = 0
                    break;
        if checker == 1:
            tri1_new.append(triangle)
    tri1_new = np.array(tri1_new)    

    #%% Detecting the Envelop of the Triangularization
    print("3/4 Detecting the Envelop")
    dictionary1 = [] ## use the diction to record all combinations of hyperplane  (n,4)
    tr1,con1 = tri1_new.shape
    for j in np.arange(tr1):
        triangle = tri1_new[j,:]
        
        for i in range(con1):
            tmp_arr = np.array([0])
            for j in range(1,n_features+1):
                tmp_arr = np.hstack((tmp_arr,triangle[(i+j)%con1]))
            hyperplane1 = tmp_arr[1:]
            dictionary1.append(hyperplane1)  
    dictionary1 = np.array(dictionary1)
    dictionary_all.append(dictionary1)    
    dictionary1 = np.sort(dictionary1,axis=1)
    
    arr, uniq_cnt = np.unique(dictionary1, axis=0, return_counts=True)
    dictionary1 = arr[uniq_cnt==1]
    #%% get predictions
    print("4/4 get predictions")
    tri_point1 = data_1[tri1_new] #(80, 3, 2)
    pred1 = np.zeros_like(y_test)
    
    for j in range(tri_point1.shape[0]):
        hulls = tri_point1[j]
        inhull_detect = in_hull_batch(hulls, X_test)
        pred_all[:,cl] = 1 * inhull_detect

    print("execute time: %.2f seconds"%(time.time()-start_time))

#%% summing predictions together
dist_max = np.zeros(pred_all.shape)  ## need the max
dist_min = np.ones(pred_all.shape)*99999  ## need the min
pred = np.ones(pred_all.shape[0],y_test.dtype)*-1

whole_test = y_test.shape[0]
outhull = 0
time_st2 = time.time()
for i in range(pred_all.shape[0]):
    if sum(pred_all[i]) == 0: ## if in no simplex
        for ind,(data_1,dictionary1) in enumerate(zip(data_all,dictionary_all)):
            dist_min[i,ind] = point_to_envelop_fast2(data_1,dictionary1,X_test[i])
        pred[i] = np.amin(np.where(dist_min[i] == np.amin(dist_min[i])))
        outhull = outhull + 1
    elif sum(pred_all[i]) == 1:
        #print(np.where(pred_all[i] == 1))
        pred[i] = np.amin(np.where(pred_all[i] == 1))
    else:
        for ind,(data_1,dictionary1) in enumerate(zip(data_all,dictionary_all)):
            if pred_all[i,ind] == 1:
                dist_max[i,ind] = point_to_envelop_fast2(data_1,dictionary1,X_test[i])
        pred[i] = np.amin(np.where(dist_max[i] == np.amax(dist_max[i])))

correct = np.mean(pred.reshape((-1,1)) == y_test.reshape((-1,1)))
print("test boundary time: %.2f second"%(time.time()-time_st2))
print("acc:",correct)
