# encoding=utf-8

import tensorflow as tf
import scipy.io as sio
import  numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 加载数据
try_data=sio.loadmat("NN/data4students.mat")

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

#1. 数据输入点
x_data=tf.placeholder(tf.float32,[None,900])
y_data=tf.placeholder(tf.float32,[None,7])

#2. 中间层和隐藏侧
h1=add_layer(x_data,900,500,tf.sigmoid)
ol=add_layer(h1,500,7,tf.sigmoid)

#3. cost function
loss=-tf.reduce_sum(y_data*tf.log(ol))

#4. 梯度下降
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#5. 初始化
init=tf.global_variables_initializer()

#6. 获得session
sess=tf.Session()

#7. 初始化所有的参数
sess.run(init)

#8. 开始运行
for i in range(1000):

    sess.run(train_step,feed_dict={x_data:train_data,y_data:train_target})
    print "h1: "+str(sess.run(h1,feed_dict={x_data:train_data,y_data:train_target}))
    print "ol: "+str(sess.run(ol,feed_dict={x_data:train_data,y_data:train_target}))
    if i % 3 ==0:
        print sess.run(loss,feed_dict={x_data:train_data,y_data:train_target})

