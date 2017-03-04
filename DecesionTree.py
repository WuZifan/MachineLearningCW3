# encoding=utf-8
import tensorflow as tf
import scipy.io as sio
import numpy as ny
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")

# 这里的784 表示一个图像有784个像素点。
# 对应到面部表情中，该值应该是45，表示有45个属性
# x = tf.placeholder(tf.float32, [None, 784]) # 表示整个数据集
x = tf.placeholder(tf.float32, [None, 45])  # 表示整个数据集

# paras
# 这个W相当于label的集合
# 对于一个ｌａｂｅｌ，其需要一种ｗｅｉｇｈｔ，对于１０个，当然需要１０个ｗｅｉｇｈｔ
# 而每个像素点都要加上ｗｅｉｇｈｔ，因此就是７８４＊１０的矩阵
# W = tf.Variable(tf.zeros([784, 10]))
W = tf.Variable(tf.zeros([45, 6]))
# 这个b中的10表示，一个图像，对应的结果是0~9中的一个，因为只有10个，因此就是10
# 所以对应到面部表情中，该值应该是6
# b = tf.Variable(tf.zeros([10]))#这个表示偏置值
b = tf.Variable(tf.zeros([6]))  # 这个表示偏置值

# 计算那个ｓｏｆｔｍａｘ的值
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 存放图片正确的ｌａｂｅｌ值
# y_ = tf.placeholder(tf.float32, [None, 10])
y_ = tf.placeholder(tf.float32, [None, 6])

# loss func　熵交叉函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init
init = tf.initialize_all_variables()

# 开启ｓｅｓｓｉｏｎ
sess = tf.Session()
sess.run(init)


def load_data(i):
    # 加载数据
    clean_data = sio.loadmat("DecisionTreeData/noisydata_students.mat")
    tdata = clean_data['x']
    ldata = clean_data['y']
    # print len(tdata)
    # print len(ldata)
    # 处理label
    label_result = []
    tdata_result = []
    for ind, label_data in enumerate(ldata):
        if ind % 10 == i:
            real_label = label_data[0]
            temp_label = [0 for i in range(6)]
            temp_label[real_label - 1] = 1
            label_result.append(temp_label)

            tdata_result.append(tdata[ind])

    ny_tdata = ny.array(tdata_result)
    ny_label = ny.array(label_result)

    return ny_tdata, ny_label


# train
# for i in range(1000):
for i in range(9):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs, batch_ys = load_data(i)
    # print batch_ys
    # print batch_xs
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# y_中存储了所有图片的正确结果，ｙ是所有的训练的结果结果
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
# 计算正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print
# 输出
test_data, test_label = load_data(9)
print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: test_data, y_: test_label}))
