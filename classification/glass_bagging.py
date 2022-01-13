import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.model_selection import train_test_split
import time
from utils import *

from sklearn.metrics import f1_score

#%% for comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#%% hyperparameters
n_samples  = 2000
n_neighbor = 14  ## for trimming
test_size  = 0.3
noise      = 0.08 ## original 0.3
n_class    = 2

n_features = 20
batch_size = 1000

time_begin = time.time()

print("loading data...")
N = n_samples  ## samples in each class

from sklearn import datasets

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data_ori = pd.read_csv('data/glass.data')

data = data_ori.to_numpy()

#%% specific handling delete class 2
X_all = data[:,:-1]
y_all = data[:,-1]

y_all[y_all == 5] = 0
y_all[y_all == 6] = 4
y_all[y_all == 7] = 5

#y_all[y_all == 4] = 3
X_all = np.array(X_all,dtype=np.float)
y_all = np.array(y_all,dtype=np.float)


n_features = X_all.shape[1]
# sample from the original data.

X_all = X_all + np.random.normal(0,0.2,X_all.shape)
X,X_test,y,y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

## auto calculate n_features and n_class
n_features = X.shape[1]
n_class = np.unique(y).shape[0]
print(X.shape)
#%% other machine learning algorithms.
y_c = y.reshape(X.shape[0])  ## change the shape of y to 1d n_samples
y_test_c = y_test.reshape(X_test.shape[0])
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
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

## random feature extraction
X_ori, X_test_ori = X,X_test ## original whole features
num_features = X_ori.shape[1]
feature_sel = 5
n_bag = (num_features-feature_sel)*4  # num_features > feature_use

pred_fea_all = np.ones((n_bag,y_test.shape[0]),y_test.dtype)

#indexes = np.random.randint(num_features, size=n_bag)

#%% start training. 
for bag in range(n_bag):
    print('**********************************************')
    print("Dealing with bag %d/%d" %(bag,n_bag-1))
    ## useful varibale
    data_all = []
    dictionary_all = []
    pred_all = np.zeros((y_test.shape[0],n_class),y_test.dtype)  
    # generate unrepeatable random indexes
    feature_use = np.random.randint(3,feature_sel+1)  ## features = 3,4,5,6
    indexes = np.random.choice(num_features, feature_use, replace=False)
    indexes = np.sort(indexes)
    #indexes = np.random.randint(num_features, size=feature_use)  ## repeat
    print(indexes)
    X,X_test = X_ori[:,indexes],X_test_ori[:,indexes]
    #print(X.shape)
    
    for cl in range(n_class):
        time_st2 = time.time()
        start_time = time.time()
        #print("Dealing with class %d/%d" %(cl,n_class-1))
        #%% 1/4 Generate Delaunay Triangularization
        #print("1/4 Generate Delaunay Triangularization")
        d_1 = np.where(y==cl) ## the id of the points, 0,1,2,3...
        #d_2,_ = np.where(y==0)
        
        data_1 = X[d_1]
        data_all.append(data_1)
        #data_2 = X[d_2]
        #print(np.where(y==1))
        
        #whole simplex
        tri1 = Delaunay(data_1)
        TriangleForTrimming1 = tri1.simplices

        
        #%% Manifold Trimming
        #print("2/4 Manifold Trimming")
        neighbors1, distances = distance_mat(data_1,n_neighbor) ## 14 original 
        #neighbors2, distances = distance_mat(data_2,n_neighbor)
        
        
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
        
        #%% Detecting the Envelop of the Triangularization
        #print("3/4 Detecting the Envelop")
        
        dictionary1 = [] ## use the diction to record all combinations of hyperplane  (n,4)
        #print(dictionary1.shape)
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
        ## sort the record and keep unique combinations:
        
        dictionary1 = np.sort(dictionary1,axis=1)
        
        arr, uniq_cnt = np.unique(dictionary1, axis=0, return_counts=True)
        dictionary1 = arr[uniq_cnt==1]

        #%% get predictions
        #print("4/4 get predictions")
        tri_point1 = data_1[tri1_new] #(80, 3, 2)
        #tri_point2 = data_2[tri2_new] #(82, 3, 2)
        
        pred1 = np.zeros_like(y_test)
        #pred2 = np.zeros_like(y_test)
        
        ## in_hull prediction v2
        for j in range(tri_point1.shape[0]):
            hulls = tri_point1[j]
            inhull_detect = in_hull_batch(hulls, X_test)
            pred_all[:,cl] = 1 * inhull_detect
        #print("execute time: %.2f seconds"%(time.time()-start_time))
    
    dist_max = np.zeros(pred_all.shape)  ## need the max
    dist_min = np.ones(pred_all.shape)*99999  ## need the min
    pred = np.ones(pred_all.shape[0],y_test.dtype)*-1
    
    whole_test = y_test.shape[0]
    outhull = 0
    #time_st2 = time.time()
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
    pred_fea_all[bag] = pred
    correct = np.mean(pred.reshape((-1,1)) == y_test.reshape((-1,1)))
    in_rate = 1 - np.float32(outhull)/whole_test
    
    print("deal with one feature setting time: %.2f second"%(time.time()-time_st2))
    print("in hull rate:", in_rate)
    print("acc:",correct)

print("==============================================")
print('Bagging predictions:')

pred_fea_all = np.array(pred_fea_all.T,dtype=np.int64)
pred_final = np.zeros(y_test.shape,y_test.dtype)*-1
length = pred_fea_all.shape[0]

for i in range(length):
    tmp_arr = pred_fea_all[i]
    pred_final[i] = np.bincount(tmp_arr).argmax()

correct = np.mean(pred_final.reshape((-1,1)) == y_test.reshape((-1,1)))
print("bagging acc:",correct)
print("whole execute time: %.2f seconds"%(time.time()-time_begin))

