
# -*- coding:utf-8 -*-

import unittest
import logging
import os

import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np

class Samples(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

    def tc1(self):
        # 构造训练集和测试集
        x_train = np.random.random((1000, 20))
        y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
        x_test = np.random.random((100, 20))
        y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

        # 1.构建模型
        model = Sequential()
        # 第一个隐藏层共64个节点，接收20个输入节点
        model.add(Dense(64, activation='relu', input_dim=20)) 
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu')) # 第二个隐藏层64个节点
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))  # 输出层10个节点

        # 2. 编译模型
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])
        # 3. 训练网络
        model.fit(x_train, y_train, epochs=20, batch_size=128)
        # 4. 评估模型
        score = model.evaluate(x_test, y_test, batch_size=128)
        # 5. 预测新数据
        result = model.predict( np.random.random((1, 20)))
        logging.info(result)
if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
