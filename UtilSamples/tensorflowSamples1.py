
# -*- coding:utf-8 -*-

# http://www.tensorfly.cn/tfdoc/get_started/introduction.html

import unittest
import logging
import os
import numpy as np
import tensorflow as tf

class Samples(unittest.TestCase):
    ''' 演示一个简单的梯度下降 
        构造一批数据，遵循Y = (0.1, 0.2)·X + 0.3，
        定义线性模型，通过梯度下降训练模型，得到w和b
    '''
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        tf.logging.set_verbosity(tf.logging.ERROR)  # 训练过程中输出相关信息
        

    def tcMain(self):
        # 构造训练数据，x是100个2维向量，y=(0.1, 0.2) · x + 0.3
        x_data = np.float32(np.random.rand(2, 100)) # 随机输入
        y_data = np.dot([0.100, 0.200], x_data) + 0.300

        # 构造线性模型
        b = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
        y = tf.matmul(W, x_data) + b

        # 定义损失函数：最小化方差
        loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5) # 学习率α=0.5
        train = optimizer.minimize(loss)

        # 初始化变量
        init = tf.initialize_all_variables()

        # 启动图 (graph)
        with tf.Session() as sess:
            sess.run(init)

            # 拟合平面
            for step in range(0, 201):
                sess.run(train)
                if step % 20 == 0:
                    evalW = sess.run(W)
                    w1 = evalW[0][0]
                    w2 = evalW[0][1]
                    logging.info('%03d: W = (%.2f, %.2f), b = %.2f' % (step, w1, w2, sess.run(b)))
        

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
