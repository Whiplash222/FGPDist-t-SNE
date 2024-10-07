# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 04:50:31 2023

@author: yue
"""

import numpy as np
import pylab
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import datetime
import random
import os
import open3d as o3d
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.neighbors import NearestNeighbors
import math
from sklearn.neighbors import KernelDensity
import time

np_config.enable_numpy_behavior()
tf.compat.v1.enable_eager_execution()
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)+1e-12
    H = np.log(sumP) + beta * np.sum(D * P) / sumP#H(Pi)
    P = P / sumP
    return H, P


def x2p(D=np.array([]), tol=1e-5, perplexity=30.0,n=379):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    
    # Initialize some variables
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)#H(P)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        # if i % 500 == 0:
        #     print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
       
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    # print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def tsne_off(D=np.array([]), no_dims=2, perplexity=30.0,n=379):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    max_iter = 100
    # initial_momentum = 0.5
    # final_momentum = 0.8
    # eta = 500#learning rate
    # min_gain = 0.01
    tf.random.set_seed(22)
    np.random.seed(22)
    Y = tf.Variable(np.random.randn(n, no_dims))

    # Compute P-values
    P = x2p(D, 1e-5, perplexity,n)

    P = P + np.transpose(P)
    P = P / np.sum(P)
    #P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    def kl_tf(Y=np.array([])):
        sum_Y = tf.reduce_sum(tf.multiply(Y,Y), 1)
        num = -2. * tf.matmul(Y, tf.transpose(Y))
        num = 1. / (1. + tf.add(tf.transpose(tf.add(num, sum_Y)), sum_Y))#低维空间Y的距离矩阵,遵循t分布
        # diag = tf.compat.v1.diag_part(num)
        num = num-tf.eye(n)

        Q = num / tf.reduce_sum(num)
        L = tf.zeros([n,n])+1e-12
        Q=tf.maximum(Q.astype(tf.double),L.astype(tf.double))
        # Q = Q+1e-12
        # lim=tf.constant(1e-12,tf.float64)
        # Q = tf.reduce_max(Q,lim)#1e-12
        C = tf.reduce_sum(P * tf.math.log(P / Q))
        return C

    # Run iterations
    tf.keras.optimizers.legacy.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,name='Adam')
    opt = tf.keras.optimizers.Adam(learning_rate=1)

    for iter in range(max_iter):
        # Compute pairwise affinities
        with tf.GradientTape() as tape:#persistent=True
            # tape.watch(Y)
            # z = tf.numpy_function(kl, [Y], tf.float32)
            # #z = kl(Y)
            z = kl_tf(Y)
        # print(z)
        dz_dY=tape.gradient(z,Y)
        # print(dz_dY)
        opt.apply_gradients([(dz_dY,Y)])
        if (iter + 1) % 10 == 0:
            print(iter)
        
    return Y

def tsne_on(D=np.array([]), no_dims=2, perplexity=30.0,n=379,Y_tmp=np.array([])):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    max_iter = 80
    # initial_momentum = 0.5
    # final_momentum = 0.8
    # eta = 500#learning rate
    # min_gain = 0.01
    tf.random.set_seed(22)
    np.random.seed(22)
    Y = tf.Variable(np.random.randn(1, no_dims))
    Y_tmp = tf.convert_to_tensor(Y_tmp, tf.float64, name='Y_tmp')

    # Compute P-values
    P = x2p(D, 1e-5, perplexity,n)

    P = P + np.transpose(P)
    P = P / np.sum(P)
    #P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)


    def kl_tf(Y=np.array([])):
        Y = tf.concat([Y_tmp,Y], axis=0)
        sum_Y = tf.reduce_sum(tf.multiply(Y,Y), 1)
        num = -2. * tf.matmul(Y, tf.transpose(Y))
        num = 1. / (1. + tf.add(tf.transpose(tf.add(num, sum_Y)), sum_Y))#低维空间Y的距离矩阵,遵循t分布
        # diag = tf.compat.v1.diag_part(num)
        num = num-tf.eye(n)
        Q = num / tf.reduce_sum(num)
        L = tf.zeros([n,n])+1e-12
        Q=tf.maximum(Q.astype(tf.double),L.astype(tf.double))
        # Q = Q+1e-12
        # lim=tf.constant(1e-12,tf.float64)
        # Q = tf.reduce_max(Q,lim)#1e-12
        C = tf.reduce_sum(P * tf.math.log(P / Q))
        return C

    # Run iterations
    tf.keras.optimizers.legacy.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,name='Adam')
    opt = tf.keras.optimizers.Adam(learning_rate=1)

    check=np.array([])
    # starttime = datetime.datetime.now()
    for iter in range(max_iter):
        # Compute pairwise affinities
        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(Y)
            # z = tf.numpy_function(kl, [Y], tf.float32)
            # #z = kl(Y)
            z = kl_tf(Y)
        # print(z)
        dz_dY=tape.gradient(z,Y)
        # print(dz_dY)
        opt.apply_gradients([(dz_dY,Y)])
        # if (iter + 1) % 10 == 0:
        #     print(iter)
        check = np.append(check,Y)
        
    return Y

def x2D_icp(insitu_num=1, D_pre=np.array([]),tau=3):
    #when new data arrives, calculate its relative distance with previous data points and update distance matrix D
    treg = o3d.pipelines.registration
    Z_c=np.zeros((insitu_num,1))
    Z_r=np.zeros((1,insitu_num+1))
    D=np.c_[D_pre,Z_c]
    D=np.r_[D,Z_r]

    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = 5

    # Initial alignment or source to target transform.
    init_source_to_target = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0]])
    
    if (insitu_num-start>=0) and (insitu_num-start<=tau-1):
        filepath='./OPENDATA/simulation/layer'+str(random_num[insitu_num])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(insitu_num-start+1)+'.csv'
        source_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                        delimiter=',',usecols=(1,2,6,7),unpack=False)
    else:
        source_np=np.loadtxt('./OPENDATA/layer'+str(random_num[insitu_num])+'.csv',dtype=np.float,skiprows=1,
                            delimiter=',',usecols=(1,2,6,7),unpack=False)
    source_np=source_np[np.where(source_np[:, 2] >= 4400)][:,[0,1,3]]

    
    scaler_train = StandardScaler()
    source_np = scaler_train.fit_transform(source_np)
    source = o3d.geometry.PointCloud()
    source.points=o3d.utility.Vector3dVector(source_np)
    source.estimate_normals()

    for i in range(insitu_num):
        if i-start>=0 and i-start<=tau-1:
            filepath='./OPENDATA/simulation/layer'+str(random_num[i])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(i-start+1)+'.csv'
            target_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                            delimiter=',',usecols=(1,2,6,7),unpack=False)
        else:
            target_np=np.loadtxt('./OPENDATA/layer'+str(random_num[i])+'.csv',dtype=np.float,skiprows=1,
                                delimiter=',',usecols=(1,2,6,7),unpack=False)
        target_np=target_np[np.where(target_np[:, 2] >= 4400)][:,[0,1,3]]


        #standardize
        # scaler_train = StandardScaler()
        target_np = scaler_train.transform(target_np)
        target = o3d.geometry.PointCloud()
        target.points=o3d.utility.Vector3dVector(target_np)
        target.estimate_normals()

        # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
        estimation = treg.TransformationEstimationPointToPlane()

        # Convergence-Criteria for Vanilla ICP
        criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                            relative_rmse=0.000001,
                                            max_iteration=50)

        # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
        save_loss_log = True

        registration_icp = treg.registration_icp(source, target, max_correspondence_distance,
                                    init_source_to_target, estimation, criteria)
        source_temp=source.transform(registration_icp.transformation)
        source_temp=np.array(source_temp.points)
        target_temp=np.array(target.points)
        source_temp=scaler_train.inverse_transform(source_temp)
        target_temp=scaler_train.inverse_transform(target_temp)
        #calculate distance
        nbrs_source = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(source_temp[:,[0,1]])
        distances_1, indices_1 = nbrs_source.kneighbors(target_temp[:,[0,1]])
        nbrs_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_temp[:,[0,1]])
        distances_2, indices_2 = nbrs_target.kneighbors(source_temp[:,[0,1]])
        distance_final1=np.sqrt(np.sum(np.square(source_temp[:,2][indices_1].flatten() - target_temp[:,2])))
        distance_final2=np.sqrt(np.sum(np.square(target_temp[:,2][indices_2].flatten() - source_temp[:,2])))
        d_icp=(distance_final1+distance_final2)/2
        D[i,insitu_num]=d_icp
        D[insitu_num,i]=D[i,insitu_num]
    return D   
    
def x2D_kdtree(insitu_num=1, D_pre=np.array([]),tau=3):
    #when new data arrives, calculate its relative distance with previous data points and update distance matrix D
    treg = o3d.pipelines.registration
    Z_c=np.zeros((insitu_num,1))
    Z_r=np.zeros((1,insitu_num+1))
    D=np.c_[D_pre,Z_c]
    D=np.r_[D,Z_r]
    
    if (insitu_num-start>=0) and (insitu_num-start<=tau-1):
        filepath='./OPENDATA/simulation/layer'+str(random_num[insitu_num])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(insitu_num-start+1)+'.csv'
        source_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                        delimiter=',',usecols=(1,2,6,7),unpack=False)
    else:
        source_np=np.loadtxt('./OPENDATA/layer'+str(random_num[insitu_num])+'.csv',dtype=np.float,skiprows=1,
                            delimiter=',',usecols=(1,2,6,7),unpack=False)
    source_np=source_np[np.where(source_np[:, 2] >= 4400)][:,[0,1,3]]

    for i in range(insitu_num):
        if i-start>=0 and i-start<=tau-1:
            filepath='./OPENDATA/simulation/layer'+str(random_num[i])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(i-start+1)+'.csv'
            target_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                            delimiter=',',usecols=(1,2,6,7),unpack=False)
        else:
            target_np=np.loadtxt('./OPENDATA/layer'+str(random_num[i])+'.csv',dtype=np.float,skiprows=1,
                                delimiter=',',usecols=(1,2,6,7),unpack=False)
        target_np=target_np[np.where(target_np[:, 2] >= 4400)][:,[0,1,3]]
        
        #calculate distance
        nbrs_source = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(source_np[:,[0,1]])
        distances_1, indices_1 = nbrs_source.kneighbors(target_np[:,[0,1]])
        nbrs_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_np[:,[0,1]])
        distances_2, indices_2 = nbrs_target.kneighbors(source_np[:,[0,1]])
        distance_final1=np.sqrt(np.sum(np.square(source_np[:,2][indices_1].flatten() - target_np[:,2])))
        distance_final2=np.sqrt(np.sum(np.square(target_np[:,2][indices_2].flatten() - source_np[:,2])))
        d_icp=(distance_final1+distance_final2)/2
        D[i,insitu_num]=d_icp
        D[insitu_num,i]=D[i,insitu_num]
    return D   

def artificial_defect(random_num=np.array([]),x=-22,y=-85):
    start=50#start time point
    #abnormal maximum
    L=2300
    L0=1500
    for tau in np.arange(3,5,1):
        for i in range(tau):
            initial_data=pd.read_csv('./OPENDATA/layer'+str(random_num[start+i])+'.csv',index_col=0)
            u=(L-L0)*math.exp(-i*i/8)+L0#abnormal value
            ab_data = initial_data
            ab_data.loc[initial_data[((abs(initial_data.iloc[:,0]-x)<=size) 
                                & (abs(initial_data.iloc[:,1]-y)<=size))].index.tolist(),"V7"] = u
            #initial_data.loc[initial_data[(abs(initial_data.iloc[:,0]-x)<=small 
            #                      and abs(initial_data.iloc[:,1]-y)<=small)].index.tolist(),"V7"] = u
            # print(len(initial_data.loc[initial_data[((abs(initial_data.iloc[:,0]-x)<=size) 
            #                     & (abs(initial_data.iloc[:,1]-y)<=size))].index.tolist(),"V7"]))
            # # print(initial_data)
            ab_data.to_csv('./OPENDATA/simulation/layer'+str(random_num[start+i])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(i+1)+'.csv',header=True, index=True)

def updateD(D=np.array([]),tau=3):
    for insitu_num in np.arange(start,start+tau,1):
        filepath='./OPENDATA/simulation/layer'+str(random_num[insitu_num])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(insitu_num-start+1)+'.csv'
        source_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                        delimiter=',',usecols=(1,2,6,7),unpack=False)
        source_np=source_np[np.where(source_np[:, 2] >= 4400)][:,[0,1,3]]
    
        for i in range(end_layer):
            if i-start>=0 and i-start<=tau-1:
                filepath='./OPENDATA/simulation/layer'+str(random_num[i])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(i-start+1)+'.csv'
                target_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                                delimiter=',',usecols=(1,2,6,7),unpack=False)
            else:
                target_np=np.loadtxt('./OPENDATA/layer'+str(random_num[i])+'.csv',dtype=np.float,skiprows=1,
                                    delimiter=',',usecols=(1,2,6,7),unpack=False)
            target_np=target_np[np.where(target_np[:, 2] >= 4400)][:,[0,1,3]]
            
            #calculate distance
            nbrs_source = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(source_np[:,[0,1]])
            distances_1, indices_1 = nbrs_source.kneighbors(target_np[:,[0,1]])
            nbrs_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_np[:,[0,1]])
            distances_2, indices_2 = nbrs_target.kneighbors(source_np[:,[0,1]])
            distance_final1=np.sqrt(np.sum(np.square(source_np[:,2][indices_1].flatten() - target_np[:,2])))
            distance_final2=np.sqrt(np.sum(np.square(target_np[:,2][indices_2].flatten() - source_np[:,2])))
            d_icp=(distance_final1+distance_final2)/2
            D[i,insitu_num]=d_icp
            D[insitu_num,i]=D[i,insitu_num]
    return D 

def updateD_icp(D=np.array([]),tau=3):
    for insitu_num in np.arange(start,start+tau,1):
        treg = o3d.pipelines.registration

        # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
        max_correspondence_distance = 5

        # Initial alignment or source to target transform.
        init_source_to_target = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0], 
                                    [0.0, 0.0, 0.0, 1.0]])
        
        filepath='./OPENDATA/simulation/layer'+str(random_num[insitu_num])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(insitu_num-start+1)+'.csv'
        source_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                        delimiter=',',usecols=(1,2,6,7),unpack=False)
        source_np=source_np[np.where(source_np[:, 2] >= 4400)][:,[0,1,3]]

        scaler_train = StandardScaler()
        source_np = scaler_train.fit_transform(source_np)
        source = o3d.geometry.PointCloud()
        source.points=o3d.utility.Vector3dVector(source_np)
        source.estimate_normals()
    
        for i in range(end_layer):
            if i-start>=0 and i-start<=tau-1:
                filepath='./OPENDATA/simulation/layer'+str(random_num[i])+'_'+str(size*10)+'_tau'+str(tau)+'_'+str(i-start+1)+'.csv'
                target_np=np.loadtxt(filepath,dtype=np.float,skiprows=1,
                                delimiter=',',usecols=(1,2,6,7),unpack=False)
            else:
                target_np=np.loadtxt('./OPENDATA/layer'+str(random_num[i])+'.csv',dtype=np.float,skiprows=1,
                                    delimiter=',',usecols=(1,2,6,7),unpack=False)
            target_np=target_np[np.where(target_np[:, 2] >= 4400)][:,[0,1,3]]

            #standardize
            # scaler_train = StandardScaler()
            target_np = scaler_train.transform(target_np)
            target = o3d.geometry.PointCloud()
            target.points=o3d.utility.Vector3dVector(target_np)
            target.estimate_normals()

            # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
            estimation = treg.TransformationEstimationPointToPlane()

            # Convergence-Criteria for Vanilla ICP
            criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                                relative_rmse=0.000001,
                                                max_iteration=50)
            
            # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
            save_loss_log = True

            registration_icp = treg.registration_icp(source, target, max_correspondence_distance,
                                        init_source_to_target, estimation, criteria)
            source_temp=source.transform(registration_icp.transformation)
            source_temp=np.array(source_temp.points)
            target_temp=np.array(target.points)
            source_temp=scaler_train.inverse_transform(source_temp)
            target_temp=scaler_train.inverse_transform(target_temp)
            #calculate distance
            nbrs_source = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(source_temp[:,[0,1]])
            distances_1, indices_1 = nbrs_source.kneighbors(target_temp[:,[0,1]])
            nbrs_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_temp[:,[0,1]])
            distances_2, indices_2 = nbrs_target.kneighbors(source_temp[:,[0,1]])
            distance_final1=np.sqrt(np.sum(np.square(source_temp[:,2][indices_1].flatten() - target_temp[:,2])))
            distance_final2=np.sqrt(np.sum(np.square(target_temp[:,2][indices_2].flatten() - source_temp[:,2])))
            d_icp=(distance_final1+distance_final2)/2
            D[i,insitu_num]=d_icp
            D[insitu_num,i]=D[i,insitu_num]
    return D   

def kde_limit(X=np.array([])):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    Y_random=kde.sample(10000,22)
    up=np.quantile(Y_random, 0.9975)
    below=np.quantile(Y_random, 0.0025)
    return up,below

goodnumber = np.load("./AM/goodnumber.npy")
goodlen=200
start=50#start time point
end_layer=60
size_type='small'

if __name__ == "__main__":
    for size in [0.7]:
        print(size*2)
        seed=1
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random_num_all = np.random.choice(goodnumber, size=goodlen,replace=False)
        D_yes=0
        ooc_record=np.zeros((0,end_layer))
        for x in np.arange(-24,-21,1):#location
            for y in np.arange(-86,-83,1):
                for abnormal_seed in np.arange(28,38,1):
                    np.random.seed(abnormal_seed)
                    ab_num = np.random.choice(goodnumber, size=end_layer-start,replace=False)
                    random_num=np.append(random_num_all[0:start],ab_num)
                    artificial_defect(random_num,x,y)#construct artificial defect
                    for tau in np.arange(3,5,1):
                        start_time = time.time()
                        print('x=',str(x))
                        print('y=',str(y))
                        print('tau=',str(tau))
                        record_now=np.zeros((1,end_layer))
                        # record_now_1=np.zeros((1,end_layer))
                        #tsne
                        D_off=np.array([[0]])
                        #off-line training (initial 30 layers)
                        n_off = 30

                        if D_yes == 0:
                            #calculation
                            for i in range(n_off-1):
                                D_off = x2D_icp(i+1, D_off,tau)
                            print('D_off done')
            
                            Y_off = tsne_off(D_off, 1, 20.0,n_off)
                            Y_tmp = Y_off

                            limit_up,limit_down=kde_limit(Y_tmp)

                            #online test (each arriving new signal)
                            D_on = D_off
                            for i in range(end_layer-n_off):
                                layer_num = i + n_off
                                print(layer_num)
                                # starttime = datetime.datetime.now()
                                D_on = x2D_icp(layer_num, D_on,tau)
                                Y_on = tsne_on(D_on ,1,20.0,layer_num+1,Y_tmp)
                                if Y_on>=limit_up or Y_on<=limit_down:
                                    record_now[0,layer_num]=layer_num+1
                                Y_tmp = np.append(Y_tmp,Y_on)
                                Y_tmp = np.reshape(Y_tmp, (-1, 1))
                                # endtime = datetime.datetime.now()
                                # print (endtime - starttime)
                                
                            pd.DataFrame(D_on).to_csv('./simulationD/simulation_D'+str(seed)+'_'+str(size*10)+'.csv',header=None, index=None)    
                            pd.DataFrame(Y_tmp).to_csv('./AM/Ytmp/Y_seed'+str(seed)+'_'+str(size*10)+'.csv',header=None, index=None)    
                            
                            D_yes =1
                            ooc_record=np.r_[ooc_record,record_now]
                            # ooc_record_1=np.r_[ooc_record_1,record_now_1]
                        else:
                            # read
                            D_tmp = pd.read_csv('./simulationD/simulation_D'+str(seed)+'_'+str(size*10)+'.csv',header=None)
                            
                            # D_tmp = pd.read_csv('./simulation_D'+str(seed)+'_'+str(size*10)+'.csv',header=None)
                            D_tmp = np.array(D_tmp)

                            D_off = D_tmp[0:30,0:30]

                            Y_tmp=np.loadtxt('./AM/Ytmp/Y_seed'+str(seed)+'_'+str(size*10)+'.csv',dtype=np.float
                                    ,delimiter=',')
                            Y_tmp = Y_tmp[0:start]
                            Y_tmp = np.reshape(Y_tmp, (-1, 1))

                            limit_up,limit_down=kde_limit(Y_tmp)

                            D=updateD_icp(D_tmp,tau)

                            #online test (each arriving new signal)
                            # D_on = D_off
                            for i in range(end_layer-start):
                                layer_num = i + start
                                # print(layer_num)
                                # starttime = datetime.datetime.now()
                                D_on=D[0:(layer_num+1),0:(layer_num+1)]
                                Y_on = tsne_on(D_on ,1,20.0,layer_num+1,Y_tmp)
                                if Y_on>=limit_up or Y_on<=limit_down:
                                    record_now[0,layer_num]=layer_num+1
                                # if abs(Y_on-ct_mean)>limit_1:
                                #     record_now_1[0,layer_num]=layer_num+1
                                Y_tmp = np.append(Y_tmp,Y_on)
                                Y_tmp = np.reshape(Y_tmp, (-1, 1))
                            ooc_record=np.r_[ooc_record,record_now]
                            
                        
                        end_time = time.time()
                        print("耗时: {:.2f}秒".format(end_time - start_time))
                    





