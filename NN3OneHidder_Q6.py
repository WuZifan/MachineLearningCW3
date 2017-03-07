# encoding: UTF-8

# limitations under the License.

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

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 900])

#
# X_mean,X_var=tf.nn.moments(X,axes=[0,1])
# scale_X=tf.Variable(tf.ones([len(train_data),900]))
# shift_X=tf.Variable(tf.zeros([len(train_data),900]))
# epsilon=0.001
# X=tf.nn.batch_normalization(X,X_mean,X_var,shift_X,scale_X,epsilon)
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 7])

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 1000
M = 500
N = 120
O = 60
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([900, L], stddev=0.2))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L])+0.1)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.2))
B2 = tf.Variable(tf.zeros([M])+0.1)
# W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.2))
# B3 = tf.Variable(tf.zeros([N])+0.1)
# W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
# B4 = tf.Variable(tf.zeros([O]))
# W5 = tf.Variable(tf.truncated_normal([O, 7], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([M, 7], stddev=0.2))
B5 = tf.Variable(tf.zeros([7])+0.1)

# The model
XX = tf.reshape(X, [-1, 900])
# Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
# Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
pkeep=tf.placeholder(tf.float32)

Y1_B=tf.matmul(XX, W1) + B1

y1_m,y1_v=tf.nn.moments(Y1_B,axes=[0])
scale1=tf.Variable(tf.ones([L]))
shift1=tf.Variable(tf.zeros([L]))
epsilon=0.001
Y1_B=tf.nn.batch_normalization(Y1_B,y1_m,y1_v,shift1,scale1,epsilon)

Y1 = tf.sigmoid(Y1_B)
Y1d=tf.nn.dropout(Y1,pkeep)

Y2_B=tf.matmul(Y1d, W2) + B2
y2_m,y2_v=tf.nn.moments(Y2_B,axes=[0])
scale2=tf.Variable(tf.ones([M]))
shift2=tf.Variable(tf.zeros([M]))
epsilon=0.001
Y2_B=tf.nn.batch_normalization(Y2_B,y2_m,y2_v,shift2,scale2,epsilon)

Y2 = tf.sigmoid(Y2_B)
Y2d=tf.nn.dropout(Y2,pkeep)
# Y3 = tf.sigmoid(tf.matmul(Y2, W3) + B3)
# Ylogits = tf.matmul(Y4, W5) + B5
# Ylogits = tf.nn.sigmoid(tf.matmul(Y4, W5) + B5)
Ylogits = tf.sigmoid(tf.matmul(Y2d, W5) + B5)

# regularization with L2
w1_l2=tf.nn.l2_loss(W1)
w2_l2=tf.nn.l2_loss(W2)
# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
beiTa_Daoshu=0.00001
cross_entropy = tf.reduce_mean(cross_entropy+(w1_l2+w2_l2)*beiTa_Daoshu)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
Y = tf.nn.softmax(Ylogits)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, learning rate = 0.003;
learning_rate=tf.placeholder(tf.float32,shape=[])
# init_rate=0.01
# learning_rate=0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# train_step = tf.train.MomentumOptimizer(learning_rate,0.95 ).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def select_data(train_data,train_target,i):
    o_data=[]
    o_target=[]
    for ind in range(len(train_data)):
        if ind % 1 ==i:
            o_data.append(train_data[ind])
            o_target.append(train_target[ind])

    o_data=np.array(o_data)
    o_target=np.array(o_target)
    return o_data,o_target

def update_learning_data1(learning_rate,i):
    if i % 500 ==0:
        return learning_rate/2
    else:
        return learning_rate

def update_learning_data2(learning_rate,i):
    if i >= 1000:
        return learning_rate*0.99
    else:
        return learning_rate

def update_learning_data3(learning_rate,i):
    if i >= 1000:
        return learning_rate*1000/i
    else:
        return learning_rate

def update_learning_data4(learning_rate, validation_accuracy, temp_valid_accuracy):
    if validation_accuracy > temp_valid_accuracy:
        return learning_rate * 1.05
    elif validation_accuracy == temp_valid_accuracy:
        return learning_rate
    else:
        return learning_rate * 0.5

temp_validation_accuracy=0
index_test=0
init_learning_rate=0.02
# while True:
print init_learning_rate # 第一次是0.01.第二次是0.05.第三次是0.001,第四次是0.1
print len(train_data)
for i in range(500*len(train_data)):


    # # index_test+=1
    [train_d,train_t]=select_data(train_data,train_target,i % 1)
    print "lenth of data:"+str(len(train_d))+" "+str(len(train_t))
    sess.run(train_step, {X: train_d, Y_: train_t,learning_rate:init_learning_rate,pkeep:0.75})
    # # l_rate=sess.run(transfor)
    # # For SGD
    # # print "cross_entropy: " + str(sess.run(cross_entropy, {X: [train_data[i % len(train_data)]], Y_: [train_target[i % len(train_data)]]}))
    # # print "cross_entropy: "+str(sess.run(cross_entropy, {X: [train_data[i % len(train_data)]], Y_: [train_target[i % len(train_data)]]}))
    # # print "accuracy: "+str(sess.run(accuracy, {X: [train_data[i % len(train_data)]], Y_: [train_target[i % len(train_data)]]}))
    train_cross_entropy=sess.run(cross_entropy, {X: train_d, Y_: train_t,learning_rate:init_learning_rate,pkeep:0.75})
    print str(i)+"cross_entropy: " + str(train_cross_entropy)

    train_accuracy=sess.run(accuracy, {X: train_d, Y_: train_t,learning_rate:init_learning_rate,pkeep:0.75})
    print str(i)+"accuracy: " + str(train_accuracy)

    valid_cross_entropy=sess.run(cross_entropy, {X: valid_data, Y_: valid_target,learning_rate:init_learning_rate,pkeep:0.75})
    print str(i) + "validation_cross_entropy: " + str(valid_cross_entropy)

    valid_accuracy=sess.run(accuracy, {X: valid_data, Y_: valid_target,learning_rate:init_learning_rate,pkeep:0.75})
    print str(i) + "validation_accuracy: " + str(valid_accuracy)

    # init_learning_rate=update_learning_data4(init_learning_rate,valid_accuracy,temp_validation_accuracy)
    # temp_validation_accuracy=valid_accuracy




