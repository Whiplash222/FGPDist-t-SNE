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

Y = pd.read_csv('./AM/sample_mean_realcase.csv',header=None)
Y=np.array(Y)
Y_off=Y[0:30,:]
# print(Y_off)
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(Y_off)
# print(kde)
Y_kde=kde.score_samples(Y_off)
print(Y_kde)
Y_random=kde.sample(10000,22)
# print(Y_random)
# up=np.quantile(Y_random, 0.995)
# below=np.quantile(Y_random, 0.005)
up=np.quantile(Y_random, 0.9975)
below=np.quantile(Y_random, 0.0025)
print(up)
print(below)

import numpy as np
import GaugeRnR
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import pandas as pd
plt.style.use('seaborn-colorblind')
end_layer=60
n_off=30
# def ooc(X=np.array([]),alpha=3):
#     record_now=np.zeros((1,end_layer))
#     ct_mean=np.array(X[0:n_off]).mean()
#     limit=alpha*np.array(X[0:n_off]).std()
#     for i in np.arange(n_off,end_layer,1):
#         if abs(X[i]-ct_mean)>limit:
#             record_now[0,i]=i+1
#     return record_now


file_name='./checkkkk_proposed.csv'
xy=pd.read_csv(file_name,header=None)


xy.columns=['FGPDist-t-SNE','Type','Layer']
# print(xy[xy['Type']==1])
xy_1=xy[xy['Type']==1]
xy_2=xy[xy['Type']==2]
xy_3=xy[xy['Type']==3]
xy_4=xy[xy['Type']==4]
#plt.figure(figsize=(20, 10), dpi=100)
#xy.plot.scatter('Layer','FGPDist-t-SNE
#', c='Type', colormap='jet')

A3=3
N=30#phase I

data=np.array(xy['FGPDist-t-SNE'])
Xtrain = data[0:N]
ucl_X   = up
cl_X    = (up+below)/2
lcl_X   = below
print(ucl_X)
print(cl_X)
print(lcl_X)

plt.figure(figsize=(40, 10), dpi=100)
plt.plot(xy['Layer'],xy['FGPDist-t-SNE'])

plt.scatter(xy_1['Layer'],xy_1['FGPDist-t-SNE'], c='red', marker='o')
plt.scatter(xy_2['Layer'],xy_2['FGPDist-t-SNE'], c='orange', marker='o')
plt.scatter(xy_3['Layer'],xy_3['FGPDist-t-SNE'], c='yellow', marker='o')
plt.scatter(xy_4['Layer'],xy_4['FGPDist-t-SNE'], c='blue', marker='o')
plt.axhline(ucl_X,color="r",label="UCL={}".format(ucl_X.round(2)))
plt.axhline(lcl_X,color="r",label="LCL={}".format(lcl_X.round(2)))
plt.axhline(cl_X,color="b",label="CL={}".format(cl_X.round(2)),linestyle='--')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Layer", fontdict={'size': 16})
plt.ylabel("FGPDist-t-SNE", fontdict={'size': 16})
plt.legend()
plt.title('FGPDist-t-SNE _ real case', fontdict={'size': 20})
plt.show()
