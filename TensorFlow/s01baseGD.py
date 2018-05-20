
# -*- coding:utf-8 -*-

# http://www.tensorfly.cn/tfdoc/get_started/introduction.html

import unittest
import logging
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Samples(unittest.TestCase):
    ''' 演示一个简单的梯度下降 
        构造一批数据，遵循Y = (0.1, 0.2)·X + 0.3，
        定义线性模型，通过梯度下降训练模型，得到w和b
    '''
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        tf.logging.set_verbosity(tf.logging.ERROR)  # 训练过程中输出相关信息
        plt.switch_backend('Qt5Agg') # 在独立窗口中弹出绘图，而不是和命令行共用一个窗口
        # plt.subplots(figsize=(5,4))  # 调整窗口大小
        plt.rcParams['figure.figsize'] = (5.0, 4.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        
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
        
class LinearRegression(Samples):
    def createData(self, m=100, D=1):
        ''' 
        m 样本数
        D 每个样本的维度
        构造训练数据 y = 0.8x + 1.2 ± epsilon
        '''
        epsilon = 0.01
        # rand(...)默认生成float64，而后面的tf.random_uniform(...)生成的是float32，
        # 故此处需要强转
        X = np.float32(np.random.rand(m, D))    
        Y = np.dot(0.8, X) + 1.2 + np.random.randn(m, D) * epsilon
        return X.T, Y.T

    def tc1(self):
        ''' 在图表中显示数据 '''
        X, Y = self.createData()

        plt.ylabel('y')
        plt.xlabel('x')
        plt.scatter(X, Y, s=1, cmap=plt.cm.Spectral)
        plt.show()

    def tc2(self):
        m = 100 # 样本数
        D = 1   # 每个样本的维度
        xData, yData = self.createData(m=m, D=D)

        # 构造线性模型 y = W·x + b
        b = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.random_uniform([1, D], minval=-1.0, maxval=1.0, dtype=tf.float32))
        y = tf.matmul(W, xData) + b

        # 定义损失函数：最小化方差 loss = ∑(y-ŷ)²
        loss = tf.reduce_mean(tf.square(y - yData))
        optimizer = tf.train.GradientDescentOptimizer(0.5) # α=0.5
        train = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for step in range(0, 101):
                sess.run(train)     # 执行梯度下降，训练模型
                if step % 20 == 0:
                    evalW = sess.run(W)
                    w = evalW[0][0]
                    logging.info('03d: W = %.2f, b = %.2f' % (step, w, sess.run(b)))

    def tcMain(self):
        m = 100 # 样本数
        D = 1   # 每个样本的维度
        xTrain, yTrain = self.createData(m=m, D=D)

        # 构造线性模型 y = W·x + b
        x = tf.placeholder("float", [1, None])
        y = tf.placeholder("float", [1, None])
        b = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.random_uniform([1, D], minval=-1.0, maxval=1.0, dtype=tf.float32))
        yHat = tf.matmul(W, x) + b

        # 定义损失函数：最小化方差 loss = ∑(y-ŷ)²
        loss = tf.reduce_mean(tf.square(yHat - y))
        optimizer = tf.train.GradientDescentOptimizer(0.5) # α=0.5
        train = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for step in range(0, 101):
                sess.run(train, feed_dict={x:xTrain, y:yTrain})     # 执行梯度下降，训练模型
                if step % 20 == 0:
                    evalW = sess.run(W)
                    w = evalW[0][0]
                    logging.info('%03d: W = %.2f, b = %.2f' % (step, w, sess.run(b)))

            # 评估模型
            xDev, yDev = self.createData(m=10, D=D)
            correct_prediction = tf.reduce_mean(tf.abs(yHat - yDev)/yDev)
            accuracy = float(1 - sess.run(correct_prediction, feed_dict={x: xDev})) * 100.
            logging.info('accuracy=%.2f%%' % accuracy)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
