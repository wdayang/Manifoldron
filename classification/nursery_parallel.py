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

import multiprocessing as mp
from multiprocessing import Pool


#%% hyperparameters
#n_samples  = 2000
n_neighbor = 14  ## for trimming
test_size  = 0.3
#noise      = 0.08 ## original 0.3
#n_class    = 2
#n_features = 20
#batch_size = 1000

#%% load dataset
print("loading data...")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data_ori = pd.read_csv('data/nursery.data')
data = data_ori.apply(le.fit_transform)
data = data.to_numpy()
X_all = data[:,:-1]
y_all = data[:,-1]
#%% specific handling delete class 2
select = np.where(y_all!=2)
X_all = X_all[select]
y_all = y_all[select]
y_all[y_all == 3] = 2
y_all[y_all == 4] = 3
n_features = X_all.shape[1]
X_all = X_all + np.random.normal(0,0.1,X_all.shape)
X,X_test,y,y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

## auto calculate n_features and n_class
n_features = X.shape[1]
n_class = np.unique(y).shape[0]
print(X.shape)
#%% add some noise
## proprecessing
n_features = X.shape[1]
n_class = np.unique(y).shape[0]
#print(n_class)
## random feature extraction
X_ori, X_test_ori = X,X_test ## original whole features
num_features = X_ori.shape[1]
feature_use = 6
#n_bag = (num_features-feature_use)*4  # num_features > feature_use
n_bag = 10

pred_fea_all = np.ones((n_bag,y_test.shape[0]),y_test.dtype)
time_st2 = time.time()



train_time = []
test_time = []
#%% start training. 

def one_round(indexes):
    #print("Dealing with meta bag %d/%d" %(bag,meta_bag-1))
    data_all = []
    dictionary_all = []
    pred_all = np.zeros((y_test.shape[0],n_class),y_test.dtype)  
    
    X,X_test = X_ori[:,indexes],X_test_ori[:,indexes]
    time_train_cls = 0
    time_test_cls = 0

    for cl in range(n_class):
        start_time = time.time()
        #print("Dealing with class %d/%d, time %.2f" %(cl,n_class-1,time.time()-time_st2))
        #%% 1/4 Generate Delaunay Triangularization
        d_1 = np.where(y==cl) ## the id of the points, 0,1,2,3.... Sometimes use '_' sometimes not. 
        data_1 = X[d_1]
        #print(data_1.shape)
        data_all.append(data_1)
        
        #print(data_1.shape)
        tri1 = Delaunay(data_1)
        TriangleForTrimming1 = tri1.simplices
        
        #%% Manifold Trimming
        #print("2/4 Manifold Trimming")
        #neighbors1, distances = distance_mat(data_1,n_neighbor) ## 14 original 
        neighbors1, distances = distance_kd(data_1,n_neighbor)
        #print(neighbors1.shape)
    
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
        
        #global train_time
        time_train_cls += time.time()-start_time

        time_tr = time.time()
        #%% get predictions
        #print("4/4 get predictions")
        tri_point1 = data_1[tri1_new] #(80, 3, 2)
        pred1 = np.zeros_like(y_test)
        
        ## in_hull prediction v2
        for j in range(tri_point1.shape[0]):
            hulls = tri_point1[j]
            inhull_detect = in_hull_batch(hulls, X_test)
            pred_all[:,cl] = 1 * inhull_detect
        #global test_time
        time_test_cls += time.time() - time_tr
    
    time_predall = time.time()
    dist_max = np.zeros(pred_all.shape)  ## need the max
    dist_min = np.ones(pred_all.shape)*99999  ## need the min
    pred = np.ones(pred_all.shape[0],y_test.dtype)*-1
    
    whole_test = y_test.shape[0]
    outhull = 0

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
            #for ind in range(n_class):
            for ind,(data_1,dictionary1) in enumerate(zip(data_all,dictionary_all)):
                if pred_all[i,ind] == 1:
                    dist_max[i,ind] = point_to_envelop_fast2(data_1,dictionary1,X_test[i])
            pred[i] = np.amin(np.where(dist_max[i] == np.amax(dist_max[i])))
        
    correct = np.mean(pred.reshape((-1,1)) == y_test.reshape((-1,1)))
    test_time = time.time() - time_predall + time_test_cls
    in_rate = 1 - np.float32(outhull)/whole_test
    one_round.counter += 1
    #print("**********************************")
    print("round %d/%d acc: %f, time %.2f"%(one_round.counter,n_bag//6+1,correct,time.time()-time_st2))
    #return pred
    return np.hstack((pred,time_train_cls,test_time))

one_round.counter = 0

#%% main function for parallel computing
if __name__ == '__main__':
    # other machine learning algorithms.
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
        y_pred = clf.predict(X_test)
        f1score = f1_score(y_test_c, y_pred, average='macro')
        print('F1 score',f1score)
    time_begin = time.time()
    #%% multiprocessing
    num_cpu = mp.cpu_count()
    num_cpu = 6
    print("number of parallel CPU:",num_cpu)
    pool = Pool(num_cpu)  ## do not go to full power
    
    #n_bag = 100
    feature_use = 6
    indexes = np.ones((n_bag,feature_use),dtype=np.int)*-1
    for i in range(n_bag):
        indexes[i] = np.random.choice(num_features, feature_use, replace=False)
    indexes = np.sort(indexes,axis=1)
    #print(indexes)
    indexes = indexes.tolist()
    
    #print("Dealing with bag %d/%d" %(bag,num_cpu-1))
    results = pool.map(one_round, [index for index in indexes])
    pool.close()
    result_all = np.array(results)
    #print(result_all.shape)
    pred_fea_all = result_all[:,:-3]  ## pred
    train_time = result_all[:,-2]     ## train time
    test_time = result_all[:,-1]      ## test time
    print(pred_fea_all.shape)
    print(pred_fea_all[:10])
    
    #%% bagging prediction
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
    f1score = f1_score(y_test, pred_final, average='macro')
    print("F1 score:",f1score)
    
    whole_time = time.time()-time_begin
    print("whole execute time: %.2f seconds"%(whole_time))
    
    train_time = np.array(train_time)
    test_time = np.array(test_time)
    train_time_all = np.sum(train_time)
    test_time_all = np.sum(test_time)
    train_time_final = train_time_all/(train_time_all+test_time_all)*whole_time  ## because parallel theme, there are overlap with the time
    test_time_final = test_time_all/(train_time_all+test_time_all)*whole_time
    
    print('Training time: %.2f seconds' %train_time_final)
    print('Inference time: %.2f seconds' %test_time_final)
    