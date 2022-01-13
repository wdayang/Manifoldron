# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:37:22 2021

@author: Dayang_Wang
"""

import numpy as np
from scipy.optimize import linprog
from numpy.linalg import norm
from numpy.linalg import lstsq
from scipy.spatial.qhull import _Qhull


#%% high dimension functions
#split data into equal chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst)-n, n):
        yield lst[i:i + n]
        
def in_hull_s(points, x):  ## for one point detection
    """
    Check whether a point is in a hull. (much faster)
    
    points: array of points in the hull (n_points,n_dim)
    x: test point (n_dim)
    """
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    try:
        lp = linprog(c, A_eq=A, b_eq=b) #,cholesky=False)   ## this part may terminate the program with infeasible solutions
        return lp.success
    except:
        return False

def in_hull_batch(points, queries):
    hull = _Qhull(b"i", points,
                  options=b"",
                  furthest_site=False,
                  incremental=False, 
                  interior_point=None)
    equations = hull.get_simplex_facet_array()[2].T
    return np.all(queries @ equations[:-1] < - equations[-1], axis=1)

def in_hull_b(p, hull):
    """
    Test if points in `p` are in `hull`  ## (slow)

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def point_plane_dist(point,w,b=1): #b = 1
    """
    Calculate the distance of a point to a hyperplane
    point: point position.
    w,b: the hyperplane defined by wx+b=0
    """
    return np.abs(np.dot(w,point)+b)/norm(w)

def point_plane_projection(normals,point):
    """
    Get the projection of a point onto hyperplane
    normals: sets of points that define the hyperplane (n_points,n_dim)
    point: the given point
    """
    normals = normals.T
    coeff = lstsq(normals, point, rcond=-1)[0]  ##  To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
    proj = np.dot(normals, coeff)
    return proj

def points_to_hyperplane(points):
    """
    Form the hyperplane from the points
    """
    b = np.ones(points.shape[0])
    return np.linalg.inv(points).dot(b)

def point_hypersegment_dist(points, x):
    """
    The smallest distance from the point to the corner of the hyper segment
    """
    smallest = 999999
    for i in range(points.shape[0]):
        tmp_dis = dist(x,points[i])
        if tmp_dis < smallest:
            smallest = tmp_dis
    return smallest

def point_to_envelop(data_1,data_2,dictionary1,dictionary2,point):
    """
    The distance of the point to the segment hyperplane
    """
    smallest1 = 999999
    for i in range(dictionary1.shape[0]):
        points = data_1[dictionary1[i]]  ## get the points of hyperplane
        #print(points.shape)
        proj = point_plane_projection(points,point)
        if in_hull_s(points, proj): ## if projection inside hyperplane
            w1 = points_to_hyperplane(points)
            dist_tmp = point_plane_dist(point,w1)
        else:
            dist_tmp = point_hypersegment_dist(points, point)
        if dist_tmp < smallest1:
            smallest1 = dist_tmp
    
    smallest2 = 999999
    for i in range(dictionary2.shape[0]):
        points = data_2[dictionary2[i]]  ## get the points of hyperplane
        proj = point_plane_projection(points,point)
        if in_hull_s(points, proj): ## if projection inside hyperplane
            w2 = points_to_hyperplane(points)
            dist_tmp = point_plane_dist(point,w2)
        else:
            dist_tmp = point_hypersegment_dist(points, point)
        if dist_tmp < smallest2:
            smallest2 = dist_tmp
            
    if smallest1 > smallest2:
        return 0
    else:
        return 1
    
def point_to_envelop_single(data_1,dictionary1,point):
    """
    The distance of the point to the segment hyperplane
    """
    smallest1 = 999999
    for i in range(dictionary1.shape[0]):
        points = data_1[dictionary1[i]]  ## get the points of hyperplane
        #print(points.shape)
        proj = point_plane_projection(points,point)
        if in_hull_s(points, proj): ## if projection inside hyperplane
            w1 = points_to_hyperplane(points)
            dist_tmp = point_plane_dist(point,w1)
        else:
            dist_tmp = point_hypersegment_dist(points, point)
        if dist_tmp < smallest1:
            smallest1 = dist_tmp
    return smallest1

def point_to_envelop_fast(data_1,dictionary1,point):
    smallest1 = 999999
    for i in range(dictionary1.shape[0]):
        points = data_1[dictionary1[i]]  ## get the points of hyperplane
        dist_tmp = point_hypersegment_dist(points, point)
        if dist_tmp < smallest1:
            smallest1 = dist_tmp
    return smallest1

def point_to_envelop_fast2(data_1,dictionary1,point):
    smallest1 = 999999
    dict_all = np.unique(np.reshape(dictionary1,(dictionary1.shape[0]*dictionary1.shape[1])))
    points = data_1[dict_all]
    dist_tmp = point_hypersegment_dist(points, point)
    
    if dist_tmp < smallest1:
        smallest1 = dist_tmp
# =============================================================================
#     for i in range(dictionary1.shape[0]):
#         points = data_1[dictionary1[i]]  ## get the points of hyperplane
#         dist_tmp = point_hypersegment_dist(points, point)
#         if dist_tmp < smallest1:
#             smallest1 = dist_tmp
# =============================================================================
    return smallest1

def point_to_simplex_regre(points,y,point):
    num,dim = points.shape
    point = point.reshape(1,dim)
    point2 = np.repeat(point,num,axis=0)
    dist = np.linalg.norm(points-point2,axis=1)
    weight = 1./dist**2
    wsum = np.sum(weight)
    weight = weight/wsum
    
    label = np.dot(y,weight)
    
    return label

def point_to_cloud_regre(data_1,dictionary1,y,point):
    dictionary1 = np.squeeze(np.array(dictionary1))
    dict_all = np.unique(np.reshape(dictionary1,(dictionary1.shape[0]*dictionary1.shape[1])))
    points = data_1[dict_all]
    y = y[dict_all]
    
    num,dim = points.shape
    point = point.reshape(1,dim)
    k = dim  ## take out k points
    
    point2 = np.repeat(point,num,axis=0)
    dist = np.linalg.norm(points-point2,axis=1)
    #dist = np.sqrt((points-point2)**2)
    dist_sort = np.sort(dist)
    sort_arg = np.argsort(dist)
    dist_sort [k:] = 0  ## 
    
    weight = 1./dist_sort[:k]**2
    wsum = np.sum(weight)
    weight = weight/wsum
    
    label = np.dot(y[sort_arg[:k]],weight)
    
    return label

def tune_bright_contrast(image,alpha,beta):
    """
    Tune the brightness and contrastness of given image
    """
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
    return new_image
    
#%% two dimension functions
def PointDeter(px,py,p0x,p0y,p1x,p1y,p2x,p2y):
    """
    Determine whether a point is in given triangle 
    px,py: test point.
    p0x...p2y: given triangle
    """
    Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
    s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
    t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)

    if s>=0 and t>=0 and 1-s-t>=0:
        return True
    else: return False
  
def dist(a, b): # L2 distance for computation
        return np.sqrt(sum((a - b)**2))

def simplex_trim(TriangleForTrimming1,data_1,n_neighbor):
    neighbors1, distances = distance_mat(data_1,n_neighbor) ## 14 original 
    #neighbors2, distances = distance_mat(data_2,n_neighbor)
    
    #TriangleForTrimming1 = tri1.simplices
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
    return tri1_new

def distance_mat(X, n_neighbors=6):
    """
    Compute the square distance matrix using Euclidean distance
    :param X: Input data, a numpy array of shape (img_height, img_width)
    :param n_neighbors: Number of nearest neighbors to consider, int
    :return: numpy array of shape (img_height, img_height), numpy array of shape (img_height, n_neighbors)
    """
    # Compute full distance matrix
    distances = np.array([[dist(p1, p2) for p2 in X] for p1 in X]) # compute full distance matrix

    # Keep only the 6 nearest neighbors, others set to 0 (= unreachable)
    neighbors = np.zeros_like(distances) # zero matrix of given matrix size
    sort_distances = np.argsort(distances, axis=1)[:, 1:n_neighbors+1] # skip the first zero same A->A,
    # ind = np.argsort(x, axis=0) # get the point index
    # np.take_along_axis(x, ind, axis=0) # get the sorted array
    for k,i in enumerate(sort_distances):
        neighbors[k,i] = distances[k,i]
    neighbors = (neighbors+neighbors.T)/2 # this step does not influence too much
    
    return neighbors, sort_distances # unkept distance set to be zero

def PointLine(x1, y1, x2, y2, x3, y3): # x3,y3 is the point
    """
    Compute the the distance of a point to finite line segment
    :param x1,y1,x2,y2: segment position
    :param x3,y3: given point
    :return: a float number distance
    """
    px = x2 - x1
    py = y2 - y1
    norm = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance
    dist = (dx*dx + dy*dy)**.5

    return dist

def CloseLine(data_1,data_2,aa1,bb1,aa2,bb2,x3,y3): ## get the closest envelop
    """
    Calculate the minimal distance from an outliere point to the boundaries and determine the class.
    """

    distli1 = np.zeros_like(bb1,dtype=float) # originally int, need to change to float
    distli2 = np.zeros_like(bb2,dtype=float)
    for k in np.arange(len(bb1)): # two consequent point here and the line
        distli1[k] = PointLine(data_1[aa1[k],0],data_1[aa1[k],1], data_1[bb1[k],0],data_1[bb1[k],1], x3,y3)
    for k in np.arange(len(bb2)):
        distli2[k] = PointLine(data_2[aa2[k],0],data_2[aa2[k],1], data_2[bb2[k],0],data_2[bb2[k],1], x3,y3)
    if min(distli1) <= min(distli2):
        return 1
    else: return 0
    
#%% metric
def show_metrics(y_true, y_score):
    # True positive
    tp = np.sum(y_true * y_score)
    # False positive
    fp = np.sum((y_true == 0) * y_score)
    # True negative
    tn = np.sum((y_true==0) * (y_score==0))
    # False negative
    fn = np.sum(y_true * (y_score==0))

    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2*tp / (2*tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    print("True positive: ", tp)
    print("False positive: ", fp)
    print("True negative: ", tn)
    print("False negative: ", fn)

    print("True positive rate (recall): ", tpr)
    print("False positive rate: ", fpr)
    print("Precision: ", precision)
    print("True negative rate: ", tnr)
    print("F1: ", f1)
    print("ROC-AUC: ", auc)
    print("MCC: ", mcc)
  

