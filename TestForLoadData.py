# encoding: UTF-8


import tensorflow as tf
import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt

tf.set_random_seed(0)

# 加载数据
try_data=sio.loadmat("data4students.mat")

input_data=try_data['datasetInputs']
target_data=try_data['datasetTargets']

# 训练的数据
train_data=input_data[0][0]
test_data=input_data[0][1]
valid_data=input_data[0][2]
# 对应的label
train_target= target_data[0][0]
test_target= target_data[0][1]
valid_target= target_data[0][2]

print np.mean(train_data[:,0])

print np.var(train_data[:,0])

for i in range(len(train_data[0])):
    mean=np.mean(train_data[:,i])
    std =np.std(train_data[:,i])
    for j in range(len(train_data)):
        train_data[j,i]=(train_data[j,i]-mean)/std

print np.mean(train_data[:,0])

print np.var(train_data[:,0])