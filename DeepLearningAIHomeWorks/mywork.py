
# -*- coding:utf-8 -*-
'''
本文件是deeplearning.ai的编程作业，需要结合相关数据文件才能执行，数据文件的根目录需赋给
CodingWorks.rootDir
'''

import logging
import unittest
import math
import sys
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

class CodingWorks(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
        self.rootDir = os.path.join(os.path.expanduser('~'), 'Documents/DeepLearningAI作业/')

    def showH5Group(self, groupObj):
        logging.info('group name:%s, shape:%s' % (groupObj.name, groupObj.shape))
        for key in groupObj.keys():
            if isinstance(groupObj[key], h5py.Group):
                self.showH5Group(groupObj[key])
            else:
                self.showH5Dataset(groupObj[key])

    def showH5Dataset(self, datasetObj):
        logging.info('dataset name:%s, shape:%s' % (datasetObj.name, datasetObj.shape))
        logging.info('value:%s' % (datasetObj.value))

    def ShowH5File(self, path):
        h5 = h5py.File(path, 'r')
        for key in h5.keys():
            if isinstance(h5[key], h5py.Group):
                self.showH5Group(h5[key])
            else:
                self.showH5Dataset(h5[key])
        return h5

class Coding1_1(CodingWorks):
    def setUp(self):
        super().setUp()
        self.datasetsDir = os.path.join(self.rootDir, 'coding1_1/datasets/')
        self.imagesDir = os.path.join(self.rootDir, 'coding1_1/images/')
        self.trainDatasetPath = os.path.join(self.datasetsDir, 'train_catvnoncat.h5')
        self.testDatasetPath = os.path.join(self.datasetsDir, 'test_catvnoncat.h5')

    def load_dataset(self):
        train_dataset = h5py.File(self.trainDatasetPath, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File(self.testDatasetPath, "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def tc1(self):
        '''
        解析训练集和测试集的数据
        '''

        h5 = self.ShowH5File(self.trainDatasetPath)
        '''
        dataset name:/list_classes, shape:(2,)
        value:[b'non-cat' b'cat']                           # 标注词典
        dataset name:/train_set_x, shape:(209, 64, 64, 3)   # 共209张图片，每张图片是64 * 64 * 3
        value:[[[[ 17  31  56] 
               [ 22  33  59]
               [ 25  35  62]
               ...,
               [  1  28  57]
               [  1  26  56]
               [  1  22  51]]
               ...
        dataset name:/train_set_y, shape:(209,)             # 共209个标注
        value:[0 0 1 ... 0 0 0]
        '''

        h5 = self.ShowH5File(self.testDatasetPath)
        '''
        dataset name:/list_classes, shape:(2,)
        value:[b'non-cat' b'cat']
        dataset name:/test_set_x, shape:(50, 64, 64, 3)     # 共50张图片
        value:[[[[158 104  83]
               [161 106  85]
               [162 107  84]
               ...,
               [170 120 100]
               [167 121 103]
               [172 127 109]]
               ...
        dataset name:/test_set_y, shape:(50,)               # 共50个标注
        value:[1 1 1 ... 1 1 0]
        '''

    def sigmoid(self, z):
        '''
        4.1 - Helper functions
        '''
        s = 1 / (1 + np.exp(-z))
        return s

    '''
    4.2 - Initializing parameters
    '''
    def initialize_with_zeros(self, dim):
        ### START CODE HERE ### (≈ 1 line of code)
        w = np.zeros((dim,1))
        b = 0
        ### END CODE HERE ###

        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        
        return w, b

    '''
    4.3 - Forward and Backward propagation
    '''
    def propagate(self, w, b, X, Y):
        m = X.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        ### START CODE HERE ### (≈ 2 lines of code)
        A = self.sigmoid(np.dot(w.T,X) + b)            # compute activation
        cost = -(np.sum(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T)))/m      # compute cost
        ### END CODE HERE ###
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        ### START CODE HERE ### (≈ 2 lines of code)
        dw = (np.dot(X,(A-Y).T))/m
        db = (np.sum(A-Y))/m
        ### END CODE HERE ###

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                 "db": db}
        
        return grads, cost

    '''
    4.4 - Optimization
    '''
    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        costs = []
        
        for i in range(num_iterations):
            # Cost and gradient calculation (≈ 1-4 lines of code)
            ### START CODE HERE ### 
            grads, cost = self.propagate(w, b, X, Y)
            ### END CODE HERE ###
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # update rule (≈ 2 lines of code)
            ### START CODE HERE ###
            w = w - learning_rate * dw
            b = b - learning_rate * db
            ### END CODE HERE ###
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w,
                  "b": b}
        
        grads = {"dw": dw,
                 "db": db}
        
        return params, grads, costs

        '''
        '''
    def predict(self, w, b, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        ### START CODE HERE ### (≈ 1 line of code)
        A = self.sigmoid(np.dot(w.T,X) + b)
        ### END CODE HERE ###

        for i in range(A.shape[1]):
            
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            ### START CODE HERE ### (≈ 4 lines of code)
            if A[:,i] >= 0.5:
                Y_prediction[:, i] = 1
            else:
                Y_prediction[:, i] = 0
            ### END CODE HERE ###
        
        assert(Y_prediction.shape == (1, m))
        
        return Y_prediction

    '''
    5 - Merge all functions into a model
    '''
    def model(self, X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        ### START CODE HERE ###
        
        # initialize parameters with zeros (≈ 1 line of code)
        w, b = self.initialize_with_zeros(X_train.shape[0]) # w元素个数等于特征数，b是一个实数，它们的值都为0

        # Gradient descent (≈ 1 line of code)
        # 学习率为learning_rate，经过num_iterations轮迭代，获得模型参数
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)
        
        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]
        
        # Predict test/train set examples (≈ 2 lines of code)
        # 验证模型在训练集和测试集上的准确率
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        ### END CODE HERE ###

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}
        
        return d

    def Main(self):
        plt.switch_backend('Qt5Agg')
        '''
        2 - Overview of the Problem set
        '''
        # 返回训练样本和测试样本的数据、标注，classes是标注对应的含义
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = self.load_dataset()
        index = 16

        msgString = 'y = %s' % (str(train_set_y[:, index]))
        msgString += ", it's a %s picture." % (classes[np.squeeze(train_set_y[:, index])].decode("utf-8"))
        logging.info(msgString)

        # 显示训练集中的一张图片
        # plt.imshow(train_set_x_orig[index])
        # plt.show()

        # 打印训练集、测试集的样本个数、像素数、通道数，以及标注数据的个数
        ### START CODE HERE ### (≈ 3 lines of code)
        m_train = train_set_x_orig.shape[0]     # 训练样本的个数
        m_test = test_set_x_orig.shape[0]       # 测试样本的个数
        num_px = train_set_x_orig.shape[1]      # 图片的宽、高像素数
        ### END CODE HERE ###
        logging.info ("Number of training examples: m_train = " + str(m_train))
        logging.info ("Number of testing examples: m_test = " + str(m_test))
        logging.info ("Height/Width of each image: num_px = " + str(num_px))
        logging.info ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        logging.info ("train_set_x shape: " + str(train_set_x_orig.shape))
        logging.info ("train_set_y shape: " + str(train_set_y.shape))
        logging.info ("test_set_x shape: " + str(test_set_x_orig.shape))
        logging.info ("test_set_y shape: " + str(test_set_y.shape))

        # 将训练和测试样本扁平化，对于每一张图，将 (64, 64, 3) 的图片转成(64*64*3, 1)
        # 对于整个训练样本，将(209, 64, 64, 3)转成(209, -1).T，注意有个转置，转置后每一列是一个样本，
        # 某列的每一行是图片的一个特征。训练集共209行（个样本），12288个特征。参见笔记2.1。
        ### START CODE HERE ### (≈ 2 lines of code)
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
        ### END CODE HERE ###

        logging.info ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
        logging.info ("train_set_y shape: " + str(train_set_y.shape))
        logging.info ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
        logging.info ("test_set_y shape: " + str(test_set_y.shape))
        logging.info ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

        # 标准化数据，让每个元素∈[0, 1]
        train_set_x = train_set_x_flatten / 255.
        test_set_x = test_set_x_flatten / 255.

        # 训练出模型
        d = self.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

        # 验证模型
        fname = os.path.join(self.imagesDir, 'my_image2.jpg')
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3)).T
        my_predicted_image = self.predict(d['w'], d['b'], my_image)
        logging.info(my_predicted_image)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()

    # cmd: python -m unittest samples.Chapter4.case4
