# encoding: UTF-8

# limitations under the License.

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

tf.set_random_seed(0)

# 加载数据
try_data = sio.loadmat("data4students.mat")

input_data = try_data['datasetInputs']
target_data = try_data['datasetTargets']

# 训练的数据
train_data = input_data[0][0]
test_data = input_data[0][1]
valid_data = input_data[0][2]
# 对应的label
train_target = target_data[0][0]
test_target = target_data[0][1]
valid_target = target_data[0][2]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 900])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 7])

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 1000
M = 200
N = 120
O = 60
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([900, L], stddev=0.2))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]) + 0.1)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.2))
B2 = tf.Variable(tf.zeros([M]) + 0.1)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.2))
B3 = tf.Variable(tf.zeros([N]) + 0.1)
# W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
# B4 = tf.Variable(tf.zeros([O]))
# W5 = tf.Variable(tf.truncated_normal([O, 7], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([N, 7], stddev=0.2))
B5 = tf.Variable(tf.zeros([7]) + 0.1)

# The model
XX = tf.reshape(X, [-1, 900])
# Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
# Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
Y1 = tf.sigmoid(tf.matmul(XX, W1) + B1)
Y2 = tf.sigmoid(tf.matmul(Y1, W2) + B2)
Y3 = tf.sigmoid(tf.matmul(Y2, W3) + B3)
# Ylogits = tf.matmul(Y4, W5) + B5
# Ylogits = tf.nn.sigmoid(tf.matmul(Y4, W5) + B5)
Ylogits = tf.nn.softmax(tf.matmul(Y3, W5) + B5)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
Y = tf.nn.softmax(Ylogits)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, learning rate = 0.003
learning_rate = 0.005
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def random_color():
    color_red = random.randint(16, 255)
    color_blue = random.randint(16, 255)
    color_green = random.randint(16, 255)
    return '#' + str(hex(color_red)[2:]) + str(hex(color_green)[2:]) + str(hex(color_blue)[2:])


def select_data(train_data, train_target, i):
    o_data = []
    o_target = []
    for ind in range(len(train_data)):
        if ind == i:
            o_data.append(train_data[ind])
            o_target.append(train_target[ind])

    o_data = np.array(o_data)
    o_target = np.array(o_target)
    return o_data, o_target

fig, ax = plt.subplots(1, 2)
fig.canvas.set_window_title('Result Evaluation')
pos = []
cross_entropies = []
accuracies = []
for i in range(20):
    [train_d, train_t] = select_data(train_data, train_target, i)
    sess.run(train_step, {X: train_data, Y_: train_target})
    temp_cross_entropy = str(sess.run(cross_entropy, {X: train_data, Y_: train_target}))
    temp_accuracy = str(sess.run(accuracy, {X: train_data, Y_: train_target}))

    print "cross_entropy: " + temp_cross_entropy
    print "accuracy: " + temp_accuracy

    pos = range(1, i + 2)
    cross_entropies.append(float(temp_cross_entropy))
    accuracies.append(float(temp_accuracy))

    ax[0].plot(pos, cross_entropies)
    ax[0].set_title("Error Function")
    ax[0].grid(True)

    ax[1].plot(pos, accuracies)
    ax[1].set_title("Accuracy")
    ax[1].grid(True)

    plt.draw()

plt.show()







