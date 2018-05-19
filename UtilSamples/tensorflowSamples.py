
# -*- coding:utf-8 -*-

import unittest
import logging
import os

import tensorflow as tf

class Samples(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
        
    def tc1(self):
        # TensorFlow库有一个默认图，此处就使用了该默认图。
        # 1. 创建三个节点：两个constant() op，一个matmul() op
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)

        # 2. 在会话中启动图
        sess = tf.Session()
        result = sess.run(product)
        logging.info(result)
        sess.close()

    def tc2(self):
        # 查看中间过程的取值
        sess = tf.InteractiveSession()
        x = tf.Variable([1.0, 2.0])
        a = tf.constant([3.0, 3.0])

        x.initializer.run()
        logging.info(x.eval())

        sub = tf.subtract(x, a)
        logging.info(sub.eval())

    def tc3(self):
        state = tf.Variable(0, name="counter")  # 创建一个变量，初始化为标量0
        one = tf.constant(1)
        new_value = tf.add(state, one)          # 创建一个实现累加的op
        update = tf.assign(state, new_value)

        # 启动图后, 变量必须先经过初始化
        init_op = tf.global_variables_initializer() # 创建一个初始化变量的 op

        with tf.Session() as sess:
            sess.run(init_op)                   # 执行初始化
            logging.info(sess.run(state))       # 打印初值
            for _ in range(3):
                sess.run(update)
                logging.info(sess.run(state))   # 打印累加值

    def tc4(self):
        ''' 在运行时指定输入实参 '''
        input1 = tf.placeholder(tf.float32) # 定义占位符形参
        input2 = tf.placeholder(tf.float32)
        output = tf.multiply(input1, input2)
        logging.info(input1)

        with tf.Session() as sess:
            logging.info(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
