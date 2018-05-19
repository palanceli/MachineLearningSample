
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

class Samples(unittest.TestCase):
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
        # 创建会话，训练模型
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            # 评估模型
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            logging.info(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
