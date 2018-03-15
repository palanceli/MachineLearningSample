
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
import sklearn
import sklearn.datasets
import sklearn.linear_model

class CodingWorks(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
        self.rootDir = os.path.join(os.path.expanduser('~'), 'Documents/DeepLearningAI作业/')
        np.random.seed(1)
        plt.switch_backend('Qt5Agg') # 在独立窗口中弹出绘图，而不是和命令行共用一个窗口

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
            grads, cost = self.propagate(w, b, X, Y) # 一次梯度下降
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
        w, b = self.initialize_with_zeros(X_train.shape[0]) # w元素个数等于特征数，b是一个实数，将他们初始化为0

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

        # 这是构成模型的全部元素，对未来有用的就是w和b
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}
        
        return d

    def Main(self):
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

class Coding1_2(CodingWorks):
    def load_planar_dataset(self):
        np.random.seed(1)
        m = 400 # 样本个数
        N = int(m/2) # 每类样本个数
        D = 2 # 每个样本的维度
        X = np.zeros((m,D)) # data matrix where each row is a single example
        Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
        a = 4 # maximum ray of the flower

        for j in range(2):
            ix = range(N * j, N * (j + 1))
            # 生成N个等差数列 + N个随机数
            t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2 # theta
            r = a * np.sin(4 * t) + np.random.randn(N) * 0.2 # radius
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j
            
        X = X.T
        Y = Y.T

        return X, Y.reshape(m)

    def plot_decision_boundary(self, model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        # np.arange三个参数分别为start, stop, step @tc1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()]) # 输入网格纵横网格线的所有交汇点@tc3
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


    def LogisticRegression(self):
        # 通过逻辑回归生成分类器，将红豆和绿豆分开
        # X是400个点，Y是400个颜色r/b值
        X, Y = self.load_planar_dataset()
        # 绘制这些点
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
        ### START CODE HERE ### (≈ 3 lines of code)
        shape_X = X.shape
        shape_Y = Y.shape
        m = shape_X[1]  # training set size
        ### END CODE HERE ###

        print ('The shape of X is: ' + str(shape_X))
        print ('The shape of Y is: ' + str(shape_Y))
        print ('I have m = %d training examples!' % (m))
        # Train the logistic regression classifier
        clf = sklearn.linear_model.LogisticRegressionCV();
        clf.fit(X.T, Y.T.ravel()); # ravel()函数是将矩阵降成1维

        # Plot the decision boundary for logistic regression
        self.plot_decision_boundary(lambda x: clf.predict(x), X, Y)
        plt.title("Logistic Regression")

        # Print accuracy
        LR_predictions = clf.predict(X.T)
        print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
               '% ' + "(percentage of correctly labelled datapoints)")


        plt.show()

    # 4.1 - Defining the neural network structure
    def layer_sizes(self, X, Y):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)
        
        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        ### START CODE HERE ### (≈ 3 lines of code)
        n_x = X.shape[0] # size of input layer
        n_h = 4
        n_y = Y.shape[0] # size of output layer
        ### END CODE HERE ###
        return (n_x, n_h, n_y)

    # 4.2 - Initialize the model's parameters
    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        
        np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
        
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        ### END CODE HERE ###
        
        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters

    # 4.3 - The Loop
    def forward_propagation(self, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###
        
        # Implement Forward Propagation to calculate A2 (probabilities)
        ### START CODE HERE ### (≈ 4 lines of code)
        Z1 = np.dot(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = sigmoid(Z2)
        ### END CODE HERE ###
        
        assert(A2.shape == (1, X.shape[1]))
        
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        
        return A2, cache

    def compute_cost(self, A2, Y, parameters):
        """
        Computes the cross-entropy cost given in equation (13)
        
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        
        Returns:
        cost -- cross-entropy cost given equation (13)
        """
        
        m = Y.shape[1] # number of example

        # Compute the cross-entropy cost
        ### START CODE HERE ### (≈ 2 lines of code)
        logprobs = np.multiply(np.log(A2),Y)+np.multiply(1-Y,np.log(1-A2))
        cost = - np.sum(logprobs) / m
        ### END CODE HERE ###
        
        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
        assert(isinstance(cost, float))
        
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
        
        # First, retrieve W1 and W2 from the dictionary "parameters".
        ### START CODE HERE ### (≈ 2 lines of code)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        ### END CODE HERE ###
            
        # Retrieve also A1 and A2 from dictionary "cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1 = cache["A1"]
        A2 = cache["A2"]
        ### END CODE HERE ###
        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1,2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
        ### END CODE HERE ###
        
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return grads
        
    def Main(self):
        # 使用双层神经网络生成分类器，将红豆和绿豆分开
        pass

    def tc1(self):
        # 验证np.meshgrid的作用
        nx, ny = (3, 2)
        x = np.linspace(0, 1, nx)   # 将x轴[0, 1]分3份，画分割线
        y = np.linspace(0, 1, ny)   # 将y轴[0, 1]分2份，画分割线
        xv, yv = np.meshgrid(x, y)  # 计算纵横分割线的交叉点

        logging.info(x)
        logging.info(y)
        logging.info(xv)
        logging.info(yv)

    def tc2(self):
        # 验证数列生成的几种方式
        x = np.linspace(0, 10, 2)   # 将[1, 10]分成2份
        logging.info(x)
        x = np.arange(0, 10, 2)     # 将[1, 10]按步长为2分成若干份
        logging.info(x)

    def tc3(self):
        # 验证np.c_的作用
        # 将方括号内的所有参数脱掉1层，依次取各参数的第1、2...n个元素，组成新的元组
        # [[1, 2, 3]], [[4, 5, 6]], 7, 8 ->[1, 2, 3, 4, 5, 6, 7, 8]
        logging.info(np.c_[np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), 7, 8])
        # [1, 2, 3], [4, 5, 6], [7, 8, 9] -> [1, 4, 7], [2, 5, 8], [3, 6, 9]
        logging.info(np.c_[np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])])

    def tc4(self):
        # 检验contourf，绘制等高线，x、y表示二维坐标，z表示每个坐标点的高度
        x, y = np.meshgrid(np.arange(0, 4, 1), np.arange(0, 4, 1))
        z = np.array([1, 1, 1, 1,
            1, 1, 1, 0,
            1, 1, 0, 0,
            1, 0, 0, 0]).reshape(x.shape)
        plt.contourf(x, y, z, cmap=plt.cm.Spectral)

        logging.info(x)
        logging.info(y)
        logging.info(z)
        plt.show()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()

    # cmd: python -m unittest samples.Chapter4.case4
