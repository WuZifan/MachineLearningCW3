# encoding=utf-8

import tensorflow as tf
import numpy as np
import scipy.io as sio

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.zeros([in_size, out_size])+0.0001)
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

train_data=input_data[0][0]
test_data=input_data[0][1]
valid_data=input_data[0][2]

train_target= target_data[0][0]
test_target= target_data[0][1]
valid_target= target_data[0][2]

min_data=[]
for d in train_data:
    temp=min(d)
    min_data.append(temp)
print min(min_data)

train_data+=abs(min(min_data))+1
min_data=[]
for d in train_data:
    temp=min(d)
    min_data.append(temp)
print min(min_data)


x_data = train_data
y_data = train_target

# 数据输入点
xs = tf.placeholder(tf.float32, [None, 900])
ys = tf.placeholder(tf.float32, [None, 7])

# 3.定义神经层：隐藏层和预测层
l1 = add_layer(xs, 900, 1000, activation_function=tf.nn.softplus)
prediction = add_layer(l1, 1000, 7, activation_function=tf.nn.softplus)

# loss=tf.add(one,-accuracy)
# 4.定义 loss 表达式
# the error between prediciton and real data
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# loss=-tf.reduce_sum(ys * tf.log(prediction))
loss=-tf.reduce_sum(ys*tf.log(prediction))
# loss = tf.reduce_mean(tf.reduce_sum(ys - prediction,reduction_indices=[1]))
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


# important step 对所有变量进行初始化
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(10000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    a=sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        # to see the step improvement
    print 'loss: '+ str(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))