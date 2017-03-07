# encoding: UTF-8

import tensorflow as tf
import numpy as np
import scipy.io as sio

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

# normalization
def normalization(nor_data):
    for i in range(len(nor_data[0])):
        mean=np.mean(nor_data[:,i])
        std =np.std(nor_data[:,i])
        for j in range(len(nor_data)):
            nor_data[j,i]=(nor_data[j,i]-mean)/std
    return nor_data

train_data=normalization(train_data)
test_data=normalization(test_data)
valid_data=normalization(valid_data)

# input layer
X = tf.placeholder(tf.float32, [None, 900])
Y_ = tf.placeholder(tf.float32, [None, 7])

# number of neurals
L = 1000
M = 500
N = 120
O = 60

# weights and bias for each layer
W1 = tf.Variable(tf.truncated_normal([900, L], stddev=1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L])+0.1)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=1))
B2 = tf.Variable(tf.zeros([M])+0.1)
W3 = tf.Variable(tf.truncated_normal([M, 7], stddev=0.2))
B3 = tf.Variable(tf.zeros([7])+0.1)

# compute the output for each layer and use dropout
pkeep=tf.placeholder(tf.float32)
    # hidderlayer 1
Y1_B=tf.matmul(X, W1) + B1
Y1 = tf.nn.relu(Y1_B)
Y1d=tf.nn.dropout(Y1,pkeep)
    # hiddenlayer 2
Y2_B=tf.matmul(Y1d, W2) + B2
Y2 = tf.nn.relu(Y2_B)
Y2d=tf.nn.dropout(Y2,pkeep)
    # outputlayer 3
Ylogits = tf.nn.softmax(tf.matmul(Y2d, W3) + B3)

neg_likehood=tf.reduce_sum(tf.log)