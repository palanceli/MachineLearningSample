
# -*- coding:utf-8 -*-

# http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import unittest
import logging
import os
import numpy as np
import tensorflow as tf

class MnistBase(unittest.TestCase):
    ''' MNIST机器学习入门
    '''
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        tf.logging.set_verbosity(tf.logging.ERROR)  # 训练过程中输出相关信息

    def loadData(self):
        '''
        60,000行训练数据集，10,000行测试数据集，每个图片为28×28=784
        '''
        return read_data_sets('MNIST_data/', one_hot=True)
        
    def tc1(self):
        mnist = self.loadData()
        # mnist = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
        logging.info(mnist.train.images.shape)          # 训练集：55,000 × 784
        logging.info(mnist.validation.images.shape)     # 开发集：5,000 × 784
        logging.info(mnist.test.images.shape)           # 测试集：10,000 × 784

    def tcMain(self):
        ''' 没有隐藏层的softmax '''
        mnist = self.loadData()
        x = tf.placeholder("float", [None, 784])
        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))
        # 构造模型，没有隐藏层
        y = tf.nn.softmax(tf.matmul(x,W) + b)
        
        # 构造损失函数
        y_ = tf.placeholder("float", [None,10])
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))

        # 构造梯度下降
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # 训练模型
            for i in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # 评估模型
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            logging.info(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

class MnistAdv(MnistBase):
    ''' 深入MNIST
    '''
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1) # 产生标准差为0.1的正态分布随机数
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        ''' 构造卷积层 '''
        # strides的含义是[batch, height, width, channels]，此处表示向右、向下的滑动步长均为1
        # 该卷积运算前后，矩阵尺寸不变
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        ''' 构造池化层 '''
        # ksize定义池化窗口大小：2×2；strides定义步长：2×2
        # 池化后大小为(n-2+1)/(2+1)
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def tcMain(self):
        mnist = self.loadData()
        sess = tf.InteractiveSession()

        x = tf.placeholder("float", shape=[None, 784]) # ?×784
        y_ = tf.placeholder("float", shape=[None, 10]) # ?×10
        # 第一层卷积：卷积核为5×5，通道数为1，共32个卷积核
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        # 为了用卷积层，把x变成一个4d向量，其第2、第3维为宽、高，最后一维为颜色通道数
        # (因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
        x_image = tf.reshape(x, [-1,28,28,1])
        # 构造本层运算
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # 第二层卷积
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # 全连接层
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        # 为了减少过拟合，在输出层之前加入dropout。用一个placeholder来代表一个神经元的
        # 输出在dropout中保持不变的概率。在训练过程中启用dropout，在测试过程中关闭dropout。
        # TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输
        # 出值的scale。所以用dropout的时候可以不用考虑scale。
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 输出层
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        sess.run(tf.initialize_all_variables())

        train_writer = tf.summary.FileWriter('summary', sess.graph)  
        # 训练网络
        for i in range(2000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                train_accuracy = float(train_accuracy) * 100.
                logging.info("step %d, training accuracy: %.2f%%"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # 验证测试集
        test_accuracy = accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        test_accuracy = float(test_accuracy) * 100.
        logging.info("test accuracy: %.2f%%"%test_accuracy)

        train_writer.close()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
