
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
import scipy.io
import tensorflow as tf
from tensorflow.python.framework import ops

class CodingWorks(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
        self.rootDir = os.path.join(os.path.expanduser('~'), 'Documents/DeepLearningAI作业/')
        np.random.seed(1)
        plt.switch_backend('Qt5Agg') # 在独立窗口中弹出绘图，而不是和命令行共用一个窗口
        plt.subplots(figsize=(5,4))  # 调整窗口大小

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

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

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

        return X, Y

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
        plt.scatter(X[0, :], X[1, :], c=y.reshape(X.shape[1]), cmap=plt.cm.Spectral)


    def LogisticRegression(self):
        # 通过逻辑回归生成分类器，将红豆和绿豆分开
        # X是400个点，Y是400个颜色r/b值
        X, Y = self.load_planar_dataset()
        # 绘制这些点
        plt.scatter(X[0, :], X[1, :], c=Y.reshape(X.shape[1]), s=40, cmap=plt.cm.Spectral)

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
        A2 = self.sigmoid(Z2)
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

    def update_parameters(self, parameters, grads, learning_rate = 1.2):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients 
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###
        
        # Retrieve each gradient from the dictionary "grads"
        ### START CODE HERE ### (≈ 4 lines of code)
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        ## END CODE HERE ###
        
        # Update rule for each parameter
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        ### END CODE HERE ###
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters

    # 4.4 - Integrate parts 4.1, 4.2 and 4.3 in nn_model()
    def nn_model(self, X, Y, n_h, num_iterations = 10000, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]
        
        # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
        ### START CODE HERE ### (≈ 5 lines of code)
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###
        logging.info('W2.shape=(%d, %d)' % W2.shape)
        
        # Loop (gradient descent)

        for i in range(0, num_iterations):
             
            ### START CODE HERE ### (≈ 4 lines of code)
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self.forward_propagation(X, parameters)
            
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.compute_cost(A2, Y, parameters)
     
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self.backward_propagation(parameters, cache, X, Y)
     
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = self.update_parameters(parameters, grads,learning_rate = 1.2)
            
            ### END CODE HERE ###
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return parameters

    # 4.5 Predictions
    def predict(self, parameters, X):
        """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (n_x, m)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        ### START CODE HERE ### (≈ 2 lines of code)
        A2, cache = self.forward_propagation(X, parameters)
        predictions = (A2 > 0.5)
        ### END CODE HERE ###
        
        return predictions

    def Main(self):
        # 使用双层神经网络生成分类器，将红豆和绿豆分开
        X, Y = self.load_planar_dataset()
        # 绘制这些点
        plt.scatter(X[0, :], X[1, :], c=Y.reshape(X.shape[1]), s=40, cmap=plt.cm.Spectral)

        parameters = self.nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

        # Plot the decision boundary
        self.plot_decision_boundary(lambda x: self.predict(parameters, x.T), X, Y)
        plt.title("Decision Boundary for hidden layer size " + str(4))

        predictions = self.predict(parameters, X)
        print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
        plt.show()

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

class Coding1_3(CodingWorks):
    def setUp(self):
        super().setUp()
        plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        self.trainDatasetPath = os.path.join(self.rootDir, 'coding1_3/datasets/train_catvnoncat.h5')
        self.testDatasetPath = os.path.join(self.rootDir, 'coding1_3/datasets/test_catvnoncat.h5')
        np.random.seed(1)

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- 每一层的节点个数[n0, n1, ..., nl]
        
        Returns:
        parameters -- {'W1':n1×n0的数组, 'b1':n1×1的数组, 'W2':n2×n1的数组, 'b2':n2×1的数组, ...}
                        W数组使用randn*0.01填充，b数组使用0填充
        """
        
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            logging.info(layer_dims[l])
            logging.info(layer_dims[l-1])
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

            
        return parameters

    def linear_forward(self, A, W, b):
        """
        实现前向传播的线性部分： Z[l] = W[l]·A[l-1] + b[l]

        Arguments:
        A -- 前一层激活函数 n[l-1] × 1
        W -- 本层W        n[l] × n [l-1]
        b -- 本层b        n[l] × 1

        Returns:
        Z -- n[l] × 1
        cache -- 缓存(A, W, b)，接下来在反向传播时还有用
        """
        
        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W,A) + b
        ### END CODE HERE ###
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache

    def sigmoid(self, Z):
        """
        实现sigmoid激活函数

        Arguments:
        Z -- numpy array of any shape
        
        Returns:
        (A, Z)
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache

    def relu(self, Z):
        """
        实现RELU激活函数

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        (A, Z)
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        实现前向传播算法
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- 前一层激活函数n[l-1] × 1
        W -- 本层W
        b -- 本层b
        activation -- 本层激活函数的名称: "sigmoid" or "relu"

        Returns:
        A -- 激活函数
        cache -- (A, W, b, Z)
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
            ### END CODE HERE ###
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            ### END CODE HERE ###

        # linear_cache : (A, W, b)
        # activation_cache : (Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
        """
        实现前向算法，根据输入层X和各层W、b计算各层Z、A。隐藏层使用RELU、输出层使用sigmoid。
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- 输入层
        parameters -- 各层W和b
        
        Returns:
        AL -- 输出层A
        caches -- 各层(Z, prev_A, W, b)
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = 'relu')
            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = 'sigmoid')
        caches.append(cache)
        ### END CODE HERE ###
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches

    def compute_cost(self, AL, Y):
        """
        实现成本函数：-Σ(Y·log(AL) + (1-Y)·log(1-AL))
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- 输出层节点
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]

        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        ### END CODE HERE ###
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost

    def linear_backward(self, dZ, cache):
        """
        根据dZ[l]，求解dW[l]、db[l] 和 dA[l-1]
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        
        ### START CODE HERE ### (≈ 3 lines of code)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis = 1, keepdims = True) / m
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db

    def relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def linear_activation_backward(self, dA, cache, activation):
        """
        根据dA[l]、(Z[L], A[l-1], W[l], b[l])及激活函数的名称，求解dW[l]、 db[l] 和 dA[l-1]
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- (Z[L], A[l-1], W[l], b[l])
        activation -- 激活函数的名称
        
        Returns:
        dA_prev -- dA[l-1]
        dW -- dW[l]
        db -- db[l]
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
        
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        ### END CODE HERE ###
        
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        ### END CODE HERE ###
            
        return parameters

    def load_data(self):
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
        ''' 查看训练集中第15张图 '''
        train_x_orig, train_y, test_x_orig, test_y, classes = self.load_data()
        index = 15
        plt.imshow(train_x_orig[index])
        msgString = "y = %s. It's a %s picture." % (str(train_y[0,index]), classes[train_y[0,index]].decode("utf-8"))
        logging.info(msgString)
        plt.show()

    def tc2(self):
        '''
        训练集和测试集的结构相同，只是训练集有209个样本，测试集有50个样本
        /list_classes, shape:(2,) 分类名称字典
        value:[b'non-cat' b'cat']
        /train_set_x, shape:(209, 64, 64, 3) 209张64×64×3的图片
        value: ...
        /train_set_y, shape:(209, ) 209个标注
        value: [0, 0, ...]
        '''

        self.ShowH5File(self.trainDatasetPath)
        self.ShowH5File(self.testDatasetPath)

    def L_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost
        
        # Parameters initialization.
        ### START CODE HERE ###
        parameters = self.initialize_parameters_deep(layers_dims)
        ### END CODE HERE ###
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = self.L_model_forward(X, parameters)
            ### END CODE HERE ###
            
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = self.compute_cost(AL, Y)
            ### END CODE HERE ###
        
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = self.L_model_backward(AL, Y, caches)
            ### END CODE HERE ###
     
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        logging.info("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p

    def Main(self):
        # 2 - Dataset
        train_x_orig, train_y, test_x_orig, test_y, classes = self.load_data()

        m_train = train_x_orig.shape[0] # 209
        num_px = train_x_orig.shape[1]  # 64
        m_test = test_x_orig.shape[0]   # 50

        logging.info ("Number of training examples: " + str(m_train))
        logging.info ("Number of testing examples: " + str(m_test))
        logging.info ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        logging.info ("train_x_orig shape: " + str(train_x_orig.shape))
        logging.info ("train_y shape: " + str(train_y.shape))
        logging.info ("test_x_orig shape: " + str(test_x_orig.shape))
        logging.info ("test_y shape: " + str(test_y.shape))

        # 转成(x(1), x(2), ..., x(m))的形式，共64*64*3行，209列
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255.
        test_x = test_x_flatten/255.

        logging.info ("train_x's shape: " + str(train_x.shape))
        logging.info ("test_x's shape: " + str(test_x.shape))

        # 5 - L-layer Neural Network
        layers_dims = [12288, 20, 7, 5, 1]
        parameters = self.L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

        pred_train = self.predict(train_x, train_y, parameters)
        pred_test = self.predict(test_x, test_y, parameters)

    def tc3(self):
        '''
        使用coding1_3中原版代码
        '''

        sys.path.append(os.path.join(self.rootDir, 'coding1_3'))
        import dnn_app_utils_v2
        def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
            """
            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
            
            Arguments:
            X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps
            
            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

            np.random.seed(1)
            costs = []                         # keep track of cost
            
            # Parameters initialization.
            ### START CODE HERE ###
            parameters = dnn_app_utils_v2.initialize_parameters_deep(layers_dims)
            ### END CODE HERE ###
            
            # Loop (gradient descent)
            for i in range(0, num_iterations):

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                ### START CODE HERE ### (≈ 1 line of code)
                AL, caches = dnn_app_utils_v2.L_model_forward(X, parameters)
                ### END CODE HERE ###
                
                # Compute cost.
                ### START CODE HERE ### (≈ 1 line of code)
                cost = dnn_app_utils_v2.compute_cost(AL, Y)
                ### END CODE HERE ###
            
                # Backward propagation.
                ### START CODE HERE ### (≈ 1 line of code)
                grads = dnn_app_utils_v2.L_model_backward(AL, Y, caches)
                ### END CODE HERE ###
         
                # Update parameters.
                ### START CODE HERE ### (≈ 1 line of code)
                parameters = dnn_app_utils_v2.update_parameters(parameters, grads, learning_rate)
                ### END CODE HERE ###
                        
                # Print the cost every 100 training example
                if print_cost and i % 100 == 0:
                    print ("Cost after iteration %i: %f" %(i, cost))
                if print_cost and i % 100 == 0:
                    costs.append(cost)
                    
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            
            return parameters


        train_x_orig, train_y, test_x_orig, test_y, classes = self.load_data()

        m_train = train_x_orig.shape[0] # 209
        num_px = train_x_orig.shape[1]  # 64
        m_test = test_x_orig.shape[0]   # 50

        # 转成(x(1), x(2), ..., x(m))的形式，共64*64*3行，209列
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255.
        test_x = test_x_flatten/255.

        # 5 - L-layer Neural Network
        layers_dims = [12288, 20, 7, 5, 1]
        parameters = dnn_app_utils_v2.initialize_parameters_deep(layers_dims)
        arameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
        
class Coding2_1_init(CodingWorks):
    def load_dataset(self):
        np.random.seed(1)
        train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
        np.random.seed(2)
        test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        test_X = test_X.T
        test_Y = test_Y.reshape((1, test_Y.shape[0]))
        return train_X, train_Y, test_X, test_Y    
        
    def tc1(self):
        '''  查看可视化的数据 '''
        train_X, train_Y, test_X, test_Y  = self.load_dataset()
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.flatten(), s=40, cmap=plt.cm.Spectral);
        plt.show()

    def initialize_parameters_zeros(self, layers_dims):
        """
        Arguments:
        layer_dims -- 各层节点个数.
        
        Returns:
        parameters -- {W1: , b1: , ... , WL: , bL:}
        """
        
        parameters = {}
        L = len(layers_dims)            # number of layers in the network
        
        for l in range(1, L):
            # parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10;
            parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            # parameters['b' + str(l)] = np.random.randn(layers_dims[l], 1) * 10;
        return parameters

    def tc2(self):
        # 打印全零的初始化数据
        parameters = self.initialize_parameters_zeros([3,2,1])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def initialize_parameters_random(self, layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
        
        np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
        parameters = {}
        L = len(layers_dims)            # integer representing the number of layers
        
        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10;
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            ### END CODE HERE ###

        return parameters

    def tc3(self):
        parameters = self.initialize_parameters_random([3, 2, 1])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def initialize_parameters_he(self, layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
        
        np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1 # integer representing the number of layers
         
        for l in range(1, L + 1):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            ### END CODE HERE ###
            
        return parameters

    def tc4(self):
        parameters = self.initialize_parameters_he([2, 4, 1])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def sigmoid(self, x):
        """
        Compute the sigmoid of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(x)
        """
        s = 1/(1+np.exp(-x))
        return s

    def relu(self, x):
        """
        Compute the relu of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- relu(x)
        """
        s = np.maximum(0,x)
        
        return s

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation (and computes the loss) presented in Figure 2.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape ()
                        b1 -- bias vector of shape ()
                        W2 -- weight matrix of shape ()
                        b2 -- bias vector of shape ()
                        W3 -- weight matrix of shape ()
                        b3 -- bias vector of shape ()
        
        Returns:
        loss -- the loss function (vanilla logistic loss)
        """
            
        # retrieve parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        z1 = np.dot(W1, X) + b1
        a1 = self.relu(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = self.relu(z2)
        z3 = np.dot(W3, a2) + b3
        a3 = self.sigmoid(z3)
        
        cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
        
        return a3, cache

    def backward_propagation(self, X, Y, cache):
        """
        Implement the backward propagation presented in figure 2.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
        cache -- cache output from forward_propagation()
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        m = X.shape[1]
        (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
        
        dz3 = 1./m * (a3 - Y)
        dW3 = np.dot(dz3, a2.T)
        db3 = np.sum(dz3, axis=1, keepdims = True)
        
        da2 = np.dot(W3.T, dz3)
        dz2 = np.multiply(da2, np.int64(a2 > 0))
        dW2 = np.dot(dz2, a1.T)
        db2 = np.sum(dz2, axis=1, keepdims = True)
        
        da1 = np.dot(W2.T, dz2)
        dz1 = np.multiply(da1, np.int64(a1 > 0))
        dW1 = np.dot(dz1, X.T)
        db1 = np.sum(dz1, axis=1, keepdims = True)
        
        gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                     "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                     "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
        
        return gradients

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of n_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters['W' + str(i)] = ... 
                      parameters['b' + str(i)] = ...
        """
        
        L = len(parameters) // 2 # number of layers in the neural networks

        # Update rule for each parameter
        for k in range(L):
            parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
            parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
            
        return parameters

    def compute_loss(self, a3, Y):
        
        """
        Implement the loss function
        
        Arguments:
        a3 -- post-activation, output of forward propagation
        Y -- "true" labels vector, same shape as a3
        
        Returns:
        loss - value of the loss function
        """
        
        m = Y.shape[1]
        logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
        loss = 1./m * np.nansum(logprobs)
        
        return loss

    def model(self, X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
        """
        实现三层神经网络: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments: 略
        
        Returns:
        parameters -- parameters learnt by the model
        """
            
        grads = {}
        costs = [] # to keep track of the loss
        m = X.shape[1] # number of examples
        layers_dims = [X.shape[0], 10, 5, 1]
        
        # Initialize parameters dictionary.
        if initialization == "zeros":
            parameters = self.initialize_parameters_zeros(layers_dims)
        elif initialization == "random":
            parameters = self.initialize_parameters_random(layers_dims)
        elif initialization == "he":
            parameters = self.initialize_parameters_he(layers_dims)

        # Loop (gradient descent)

        for i in range(0, num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            a3, cache = self.forward_propagation(X, parameters)
            
            # Loss
            cost = self.compute_loss(a3, Y)

            # Backward propagation.
            grads = self.backward_propagation(X, Y, cache)
            
            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)
            
            # Print the loss every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
                costs.append(cost)
                
        # plot the loss
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  n-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        p = np.zeros((1,m), dtype = np.int)
        
        # Forward propagation
        a3, caches = self.forward_propagation(X, parameters)
        
        # convert probas to 0/1 predictions
        for i in range(0, a3.shape[1]):
            if a3[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        # print results
        print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
        
        return p

    def tc5(self):
        train_X, train_Y, test_X, test_Y = self.load_dataset()
        parameters = self.model(train_X, train_Y, initialization = "zeros")
        print ("On the train set:")
        predictions_train = self.predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = self.predict(test_X, test_Y, parameters)

        plt.title("Model with large random initialization")
        axes = plt.gca()
        axes.set_xlim([-1.5,1.5])
        axes.set_ylim([-1.5,1.5])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

    def plot_decision_boundary(self, model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)
        plt.show()

    def predict_dec(self, parameters, X):
        """
        Used for plotting decision boundary.
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (m, K)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        
        # Predict using forward propagation and a classification threshold of 0.5
        a3, cache = self.forward_propagation(X, parameters)
        predictions = (a3>0.5)
        return predictions

    def tc6(self):
        plt.subplots(figsize=(5,4))
        train_X, train_Y, test_X, test_Y = self.load_dataset()
        parameters = self.model(train_X, train_Y, initialization = "random")
        print ("On the train set:")
        predictions_train = self.predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = self.predict(test_X, test_Y, parameters)

        plt.subplots(figsize=(5,4))
        plt.title("Model with large random initialization")
        axes = plt.gca()
        axes.set_xlim([-1.5,1.5])
        axes.set_ylim([-1.5,1.5])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

    def tc7(self):
        plt.subplots(figsize=(5,4))
        train_X, train_Y, test_X, test_Y = self.load_dataset()
        parameters = self.model(train_X, train_Y, initialization = "he")
        print ("On the train set:")
        predictions_train = self.predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = self.predict(test_X, test_Y, parameters)

        plt.subplots(figsize=(5,4))
        plt.title("Model with He initialization")
        axes = plt.gca()
        axes.set_xlim([-1.5,1.5])
        axes.set_ylim([-1.5,1.5])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

class Coding2_1_reg(CodingWorks):
    def setUp(self):
        super().setUp()
        self.dataPath = os.path.join(self.rootDir, 'coding2_1', 'datasets/data.mat')
        
    def load_2D_dataset(self):
        data = scipy.io.loadmat(self.dataPath)
        train_X = data['X'].T
        train_Y = data['y'].T
        test_X = data['Xval'].T
        test_Y = data['yval'].T

        return train_X, train_Y, test_X, test_Y

    def tc1(self):
        ''' 数据可视化 '''
        plt.subplots(figsize=(5,4))
        train_X, train_Y, test_X, test_Y = self.load_2D_dataset()
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.flatten(), s=40, cmap=plt.cm.Spectral);
        plt.show()

    def initialize_parameters(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        b1 -- bias vector of shape (layer_dims[l], 1)
                        Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                        bl -- bias vector of shape (1, layer_dims[l])
                        
        Tips:
        - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
        This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
        - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
        """
        
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims) # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
            assert(parameters['W' + str(l)].shape == layer_dims[l], 1)

            
        return parameters

    def sigmoid(self, x):
        """
        Compute the sigmoid of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(x)
        """
        s = 1/(1+np.exp(-x))
        return s

    def relu(self, x):
        """
        Compute the relu of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- relu(x)
        """
        s = np.maximum(0,x)
        
        return s

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation (and computes the loss) presented in Figure 2.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape ()
                        b1 -- bias vector of shape ()
                        W2 -- weight matrix of shape ()
                        b2 -- bias vector of shape ()
                        W3 -- weight matrix of shape ()
                        b3 -- bias vector of shape ()
        
        Returns:
        loss -- the loss function (vanilla logistic loss)
        """
            
        # retrieve parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = self.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.relu(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = self.sigmoid(Z3)
        
        cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
        
        return A3, cache

    def compute_cost(self, a3, Y):
        """
        Implement the cost function
        
        Arguments:
        a3 -- post-activation, output of forward propagation
        Y -- "true" labels vector, same shape as a3
        
        Returns:
        cost - value of the cost function
        """
        m = Y.shape[1]
        
        logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
        cost = 1./m * np.nansum(logprobs)
        
        return cost

    def backward_propagation(self, X, Y, cache):
        """
        Implement the backward propagation presented in figure 2.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
        cache -- cache output from forward_propagation()
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1./m * np.dot(dZ2, A1.T)
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T)
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                     "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                     "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(i)] = Wi
                        parameters['b' + str(i)] = bi
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(i)] = dWi
                        grads['db' + str(i)] = dbi
        learning_rate -- the learning rate, scalar.
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        
        n = len(parameters) // 2 # number of layers in the neural networks

        # Update rule for each parameter
        for k in range(n):
            parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
            parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
            
        return parameters

    def compute_cost_with_regularization(self, A3, Y, parameters, lambd):
        """
        Implement the cost function with L2 regularization. See formula (2) above.
        
        Arguments:
        A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model
        
        Returns:
        cost - value of the regularized loss function (formula (2))
        """
        m = Y.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        W3 = parameters["W3"]
        
        cross_entropy_cost = self.compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
        
        ### START CODE HERE ### (approx. 1 line)
        L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) * lambd / (2 * m)
        ### END CODER HERE ###
        
        cost = cross_entropy_cost + L2_regularization_cost
        
        return cost

    def backward_propagation_with_regularization(self, X, Y, cache, lambd):
        """
        Implements the backward propagation of our baseline model to which we added an L2 regularization.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        
        ### START CODE HERE ### (approx. 1 line)
        dW3 = 1./m * np.dot(dZ3, A2.T) + lambd * W3 / m
        ### END CODE HERE ###
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        ### START CODE HERE ### (approx. 1 line)
        dW2 = 1./m * np.dot(dZ2, A1.T) + lambd * W2 / m
        ### END CODE HERE ###
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        ### START CODE HERE ### (approx. 1 line)
        dW1 = 1./m * np.dot(dZ1, X.T) + lambd * W1 / m
        ### END CODE HERE ###
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                     "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                     "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients

    def forward_propagation_with_dropout(self, X, parameters, keep_prob = 0.5):
        """
        Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
        
        Arguments:
        X -- input dataset, of shape (2, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (20, 2)
                        b1 -- bias vector of shape (20, 1)
                        W2 -- weight matrix of shape (3, 20)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)
        keep_prob - probability of keeping a neuron active during drop-out, scalar
        
        Returns:
        A3 -- last activation value, output of the forward propagation, of shape (1,1)
        cache -- tuple, information stored for computing the backward propagation
        """
        
        np.random.seed(1)
        
        # retrieve parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = self.relu(Z1)
        ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
        D1 = np.random.rand(A1.shape[0], A1.shape[1])     # Step 1: initialize matrix D1 = np.random.rand(..., ...)
        D1 = (D1 < keep_prob)                             # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
        A1 = np.multiply(A1, D1)                          # Step 3: shut down some neurons of A1
        A1 = A1 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        Z2 = np.dot(W2, A1) + b2
        A2 = self.relu(Z2)
        ### START CODE HERE ### (approx. 4 lines)
        D2 = np.random.rand(A2.shape[0], A2.shape[1])     # Step 1: initialize matrix D2 = np.random.rand(..., ...)
        D2 = (D2 < keep_prob)                             # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
        A2 = np.multiply(A2, D2)                          # Step 3: shut down some neurons of A2
        A2 = A2 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        Z3 = np.dot(W3, A2) + b3
        A3 = self.sigmoid(Z3)
        
        cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
        
        return A3, cache

    def backward_propagation_with_dropout(self, X, Y, cache, keep_prob):
        """
        Implements the backward propagation of our baseline model to which we added dropout.
        
        Arguments:
        X -- input dataset, of shape (2, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation_with_dropout()
        keep_prob - probability of keeping a neuron active during drop-out, scalar
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        
        m = X.shape[1]
        (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        dA2 = np.dot(W3.T, dZ3)
        ### START CODE HERE ### (≈ 2 lines of code)
        dA2 = np.multiply(dA2, D2)              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
        dA2 = dA2 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1./m * np.dot(dZ2, A1.T)
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1 = np.multiply(dA1, D1)              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
        dA1 = dA1 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T)
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                     "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                     "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients

    def model(self, X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
        """
        Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
        learning_rate -- learning rate of the optimization
        num_iterations -- number of iterations of the optimization loop
        print_cost -- If True, print the cost every 10000 iterations
        lambd -- regularization hyperparameter, scalar
        keep_prob - probability of keeping a neuron active during drop-out, scalar.
        
        Returns:
        parameters -- parameters learned by the model. They can then be used to predict.
        """
            
        grads = {}
        costs = []                            # to keep track of the cost
        m = X.shape[1]                        # number of examples
        layers_dims = [X.shape[0], 20, 3, 1]
        
        # Initialize parameters dictionary.
        parameters = self.initialize_parameters(layers_dims)

        # Loop (gradient descent)

        for i in range(0, num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            if keep_prob == 1:
                a3, cache = self.forward_propagation(X, parameters)
            elif keep_prob < 1:
                a3, cache = self.forward_propagation_with_dropout(X, parameters, keep_prob)

            # Cost function
            if lambd == 0:
                cost = self.compute_cost(a3, Y)
            else:
                cost = self.compute_cost_with_regularization(a3, Y, parameters, lambd)
                
            # Backward propagation.
            assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                                # but this assignment will only explore one at a time
            if lambd == 0 and keep_prob == 1:
                grads = self.backward_propagation(X, Y, cache)
            elif lambd != 0:
                grads = self.backward_propagation_with_regularization(X, Y, cache, lambd)
            elif keep_prob < 1:
                grads = self.backward_propagation_with_dropout(X, Y, cache, keep_prob)
            
            if i == 10:
                (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

                logging.info('Z1.shape:(%d, %d)' % Z1.shape)
                logging.info('D1.shape:(%d, %d)' % D1.shape)
                logging.info('A1.shape:(%d, %d)' % A1.shape)
                logging.info('W1.shape:(%d, %d)' % W1.shape)
                logging.info('b1.shape:(%d, %d)' % b1.shape)
                logging.info('')

                logging.info('Z2.shape:(%d, %d)' % Z2.shape)
                logging.info('D2.shape:(%d, %d)' % D2.shape)
                logging.info('A2.shape:(%d, %d)' % A2.shape)
                logging.info('W2.shape:(%d, %d)' % W2.shape)
                logging.info('b2.shape:(%d, %d)' % b2.shape)
                logging.info('')

                logging.info('Z3.shape:(%d, %d)' % Z3.shape)
                logging.info('A3.shape:(%d, %d)' % A3.shape)
                logging.info('W3.shape:(%d, %d)' % W3.shape)
                logging.info('b3.shape:(%d, %d)' % b3.shape)
                logging.info('')

                logging.info('dZ3.shape:(%d, %d)' % grads['dZ3'].shape)
                logging.info('dW3.shape:(%d, %d)' % grads['dW3'].shape)
                logging.info('db3.shape:(%d, %d)' % grads['db3'].shape)
                logging.info('dA2.shape:(%d, %d)' % grads['dA2'].shape)
                logging.info('')

                logging.info('dZ2.shape:(%d, %d)' % grads['dZ2'].shape)
                logging.info('dW2.shape:(%d, %d)' % grads['dW2'].shape)
                logging.info('db2.shape:(%d, %d)' % grads['db2'].shape)
                logging.info('dA1.shape:(%d, %d)' % grads['dA1'].shape)
                logging.info('')

                logging.info('dZ1.shape:(%d, %d)' % grads['dZ1'].shape)
                logging.info('dW1.shape:(%d, %d)' % grads['dW1'].shape)
                logging.info('db1.shape:(%d, %d)' % grads['db1'].shape)

            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)
            
            # Print the loss every 10000 iterations
            if print_cost and i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
            if print_cost and i % 1000 == 0:
                costs.append(cost)
        
        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  n-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        p = np.zeros((1,m), dtype = np.int)
        
        # Forward propagation
        a3, caches = self.forward_propagation(X, parameters)
        
        # convert probas to 0/1 predictions
        for i in range(0, a3.shape[1]):
            if a3[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        # print results

        #print ("predictions: " + str(p[0,:]))
        #print ("true labels: " + str(y[0,:]))
        print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
        
        return p

    def plot_decision_boundary(self, model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)
        plt.show()
            
    def predict_dec(self, parameters, X):
        """
        Used for plotting decision boundary.
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (m, K)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        
        # Predict using forward propagation and a classification threshold of 0.5
        a3, cache = self.forward_propagation(X, parameters)
        predictions = (a3>0.5)
        return predictions

    def tc2(self):
        ''' 不是用任何正则化 '''
        plt.subplots(figsize=(5,4))
        train_X, train_Y, test_X, test_Y = self.load_2D_dataset()
        parameters = self.model(train_X, train_Y)
        logging.info ("On the training set:")
        predictions_train = self.predict(train_X, train_Y, parameters)
        logging.info ("On the test set:")
        predictions_test = self.predict(test_X, test_Y, parameters)

        plt.subplots(figsize=(5,4))
        plt.title("Model without regularization")
        axes = plt.gca()
        axes.set_xlim([-0.75,0.40])
        axes.set_ylim([-0.75,0.65])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

    def tc3(self):
        ''' 使用L2正则化 '''
        plt.subplots(figsize=(5,4))
        train_X, train_Y, test_X, test_Y = self.load_2D_dataset()
        parameters = self.model(train_X, train_Y, lambd = 0.7)
        print ("On the train set:")
        predictions_train = self.predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = self.predict(test_X, test_Y, parameters)

        plt.subplots(figsize=(5,4))
        plt.title("Model without regularization")
        axes = plt.gca()
        axes.set_xlim([-0.75,0.40])
        axes.set_ylim([-0.75,0.65])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

    def tc4(self):
        ''' 使用Dropout正则化 '''
        plt.subplots(figsize=(5,4))
        train_X, train_Y, test_X, test_Y = self.load_2D_dataset()
        parameters = self.model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

        print ("On the train set:")
        predictions_train = self.predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = self.predict(test_X, test_Y, parameters)

        plt.subplots(figsize=(5,4))
        plt.title("Model without regularization")
        axes = plt.gca()
        axes.set_xlim([-0.75,0.40])
        axes.set_ylim([-0.75,0.65])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

class Coding2_1_gc(CodingWorks):

    def sigmoid(self, x):
        """
        Compute the sigmoid of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(x)
        """
        s = 1/(1+np.exp(-x))
        return s

    def relu(self, x):
        """
        Compute the relu of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- relu(x)
        """
        s = np.maximum(0,x)
        
        return s

    def forward_propagation_n(self, X, Y, parameters):
        """
        Implements the forward propagation (and computes the cost) presented in Figure 3.
        
        Arguments:
        X -- training set for m examples
        Y -- labels for m examples 
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (5, 4)
                        b1 -- bias vector of shape (5, 1)
                        W2 -- weight matrix of shape (3, 5)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)
        
        Returns:
        cost -- the cost function (logistic cost for one example)
        """
        
        # retrieve parameters
        m = X.shape[1]
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]

        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = self.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.relu(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = self.sigmoid(Z3)

        # Cost
        logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
        cost = 1./m * np.sum(logprobs)
        
        cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
        
        return cost, cache
    
    def backward_propagation_n(self, X, Y, cache):
        """
        Implement the backward propagation presented in figure 2.
        
        Arguments:
        X -- input datapoint, of shape (input size, 1)
        Y -- true "label"
        cache -- cache output from forward_propagation_n()
        
        Returns:
        gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
        """
        
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1./m * np.dot(dZ2, A1.T)
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T)
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                    "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                    "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients

    def gradient_check_n_test_case(self): 
        np.random.seed(1)
        x = np.random.randn(4,3)
        y = np.array([1, 1, 0])
        W1 = np.random.randn(5,4) 
        b1 = np.random.randn(5,1) 
        W2 = np.random.randn(3,5) 
        b2 = np.random.randn(3,1) 
        W3 = np.random.randn(1,3) 
        b3 = np.random.randn(1,1) 
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

        return x, y, parameters
        
    def vector_to_dictionary(self, theta):
        """
        Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
        """
        parameters = {}
        parameters["W1"] = theta[:20].reshape((5,4))
        parameters["b1"] = theta[20:25].reshape((5,1))
        parameters["W2"] = theta[25:40].reshape((3,5))
        parameters["b2"] = theta[40:43].reshape((3,1))
        parameters["W3"] = theta[43:46].reshape((1,3))
        parameters["b3"] = theta[46:47].reshape((1,1))

        return parameters

    def gradients_to_vector(self, gradients):
        """
        Roll all our gradients dictionary into a single vector satisfying our specific required shape.
        """
        
        count = 0
        for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
            # flatten parameter
            new_vector = np.reshape(gradients[key], (-1,1))
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta

    def dictionary_to_vector(self, parameters):
        """
        Roll all our parameters dictionary into a single vector satisfying our specific required shape.
        """
        keys = []
        count = 0
        for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            
            # flatten parameter
            new_vector = np.reshape(parameters[key], (-1,1))
            keys = keys + [key]*new_vector.shape[0]
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta, keys

    def gradient_check_n(self, parameters, gradients, X, Y, epsilon = 1e-7):
        """
        Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
        
        Arguments:
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
        grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
        x -- input datapoint, of shape (input size, 1)
        y -- true "label"
        epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
        
        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
        """
        
        # Set-up variables
        # 将W1 b1 W2 b2 W3 b3扁平化为47×1
        parameters_values, _ = self.dictionary_to_vector(parameters)
        # logging.info(parameters_values)
        # logging.info(parameters_values.shape)

        # 将dW1 db1 dW2 db2 dW3 db3扁平化为47×1
        grad = self.gradients_to_vector(gradients)
        # logging.info(grad)
        # logging.info(grad.shape)

        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        
        # Compute gradapprox
        for i in range(num_parameters):
            
            # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
            # "_" is used because the function you have to outputs two parameters but we only care about the first one
            ### START CODE HERE ### (approx. 3 lines)
            thetaplus = np.copy(parameters_values)                            # Step 1
            thetaplus[i][0] = thetaplus[i][0] + epsilon                       # Step 2
            J_plus[i], _ = self.forward_propagation_n(X, Y, self.vector_to_dictionary(thetaplus))                                   # Step 3
            ### END CODE HERE ###
            
            # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
            ### START CODE HERE ### (approx. 3 lines)
            thetaminus = np.copy(parameters_values)                           # Step 1
            thetaminus[i][0] = thetaminus[i][0] - epsilon                     # Step 2        
            J_minus[i], _ = self.forward_propagation_n(X, Y, self.vector_to_dictionary(thetaminus))                                  # Step 3
            ### END CODE HERE ###
            
            # Compute gradapprox[i]
            ### START CODE HERE ### (approx. 1 line)
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
            ### END CODE HERE ###
        
        # Compare gradapprox to backward propagation gradients by computing difference.
        ### START CODE HERE ### (approx. 1 line)
        numerator = np.linalg.norm(grad - gradapprox)                         # Step 1'
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)       # Step 2'
        difference = numerator / denominator                                  # Step 3'
        ### END CODE HERE ###

        if difference > 1e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
        
        return difference

    def tc1(self):
        # 随机生成X(4×3)、Y(1×3)、W1(5×4)、b1(5×1)、W2(3×5)、b2(3×1)、W3(1×3)、b3(1×1)
        X, Y, parameters = self.gradient_check_n_test_case()
        logging.info('X:%s' % X)
        logging.info('Y:%s' % Y)

        # 完成正向传播算法，返回cost, (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
        cost, cache = self.forward_propagation_n(X, Y, parameters)
        # logging.info('Z1:%s' % cache[0])
        # logging.info('A1:%s' % cache[1])
        # logging.info('W1:%s' % cache[2])
        # logging.info('b1:%s' % cache[3])
        # logging.info('Z2:%s' % cache[4])
        # logging.info('A2:%s' % cache[5])
        # logging.info('W2:%s' % cache[6])
        # logging.info('b2:%s' % cache[7])
        # logging.info('Z3:%s' % cache[8])
        # logging.info('A3:%s' % cache[9])
        # logging.info('W3:%s' % cache[10])
        # logging.info('b3:%s' % cache[11])
        # 完成反向传播算法，返回{dZ3, dW3, db3, dA2, dZ2, dW2, db2, dA1, dZ1, dW1, db1}
        gradients = self.backward_propagation_n(X, Y, cache)
        # logging.info('dZ3:%s' % gradients['dZ3'])
        # logging.info('dW3:%s' % gradients['dW3'])
        # logging.info('db3:%s' % gradients['db3'])
        # logging.info('dA2:%s' % gradients['dA2'])
        # logging.info('dZ2:%s' % gradients['dZ2'])
        # logging.info('dW2:%s' % gradients['dW2'])
        # logging.info('db2:%s' % gradients['db2'])
        # logging.info('dA1:%s' % gradients['dA1'])
        # logging.info('dZ1:%s' % gradients['dZ1'])
        # logging.info('dW1:%s' % gradients['dW1'])
        # logging.info('db1:%s' % gradients['db1'])
        difference = self.gradient_check_n(parameters, gradients, X, Y)

class Coding2_2(CodingWorks):
    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # 训练集的样本个数
        mini_batches = []
            
        # Step 1: 打乱样本顺序
        permutation = list(np.random.permutation(m)) # 返回将range(m)洗牌后的序列
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))
        # logging.info(permutation)

        # Step 2: 将样本分成mini-batch子集
        num_complete_minibatches = math.floor(m/mini_batch_size) # floor返回下折整数
        # logging.info('m=%d, mini_batch_size=%d, num_complete_minibatches=%d' % (m, mini_batch_size, num_complete_minibatches))
        for k in range(0, num_complete_minibatches): 
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, (k+1) * mini_batch_size :]
            mini_batch_Y = shuffled_Y[:, (k+1) * mini_batch_size :]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
        
    def random_mini_batches_test_case(self):
        np.random.seed(1)
        mini_batch_size = 64
        X = np.random.randn(12288, 148)
        Y = np.random.randn(1, 148) < 0.5
        return X, Y, mini_batch_size

    def tc1(self):
        ''' 验证将样本分割为mini-batch子集 '''
        X_assess, Y_assess, mini_batch_size = self.random_mini_batches_test_case()
        mini_batches = self.random_mini_batches(X_assess, Y_assess, mini_batch_size)

        print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
        print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
        print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
        print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
        print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
        print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
        print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

    def initialize_velocity(self, parameters):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        
        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """
        
        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        
        # Initialize velocity
        for l in range(L):
            ### START CODE HERE ### (approx. 2 lines)
            v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
            v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0], parameters["b" + str(l+1)].shape[1]))
            ### END CODE HERE ###
            
        return v
            
    def update_parameters_with_momentum(self, parameters, grads, v, beta, learning_rate):
        """
        Update parameters using Momentum
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- python dictionary containing the current velocity:
                        v['dW' + str(l)] = ...
                        v['db' + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        learning_rate -- the learning rate, scalar
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- python dictionary containing your updated velocities
        """

        L = len(parameters) // 2 # number of layers in the neural networks
        
        # Momentum update for each parameter
        for l in range(L):
            
            ### START CODE HERE ### (approx. 4 lines)
            # compute velocities
            v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]
            # update parameters
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
            ### END CODE HERE ###
            
        return parameters, v

    def update_parameters_with_momentum_test_case(self):
        np.random.seed(1)
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)

        dW1 = np.random.randn(2,3)
        db1 = np.random.randn(2,1)
        dW2 = np.random.randn(3,3)
        db2 = np.random.randn(3,1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        v = {'dW1': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
            [ 0.]]), 'db2': np.array([[ 0.],
            [ 0.],
            [ 0.]])}
        return parameters, grads, v
    
    def tc2(self):
        # 构造parameters:W1, b1, W2, b2 全为随机数
        # grads:dW1, db1, dW2, db2  全为随机数
        # v:vdW1, vdW2, vdb1, vdb2  全为0
        parameters, grads, v = self.update_parameters_with_momentum_test_case()

        # 遍历每一层执行
        # vdW = β·vdW + (1-β)·dW    W = W - α·vdW
        # vdb = β·vdb + (1-β)·db    b = b - α·vdb
        parameters, v = self.update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
        print("v[\"dW1\"] = " + str(v["dW1"]))
        print("v[\"db1\"] = " + str(v["db1"]))
        print("v[\"dW2\"] = " + str(v["dW2"]))
        print("v[\"db2\"] = " + str(v["db2"]))

    def initialize_adam(self, parameters) :
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters["W" + str(l)] = Wl
                        parameters["b" + str(l)] = bl
        
        Returns: 
        v -- python dictionary that will contain the exponentially weighted average of the gradient.
                        v["dW" + str(l)] = ...
                        v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                        s["dW" + str(l)] = ...
                        s["db" + str(l)] = ...

        """
        
        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}
        
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
        ### START CODE HERE ### (approx. 4 lines)
            v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
            s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        ### END CODE HERE ###
        
        return v, s

    def initialize_adam_test_case(self):
        np.random.seed(1)
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def tc3(self):
        # 构造参数W1, b1, W2, b2
        parameters = self.initialize_adam_test_case()
        # 初始化参数vdW, vdb, sdW, sdb
        v, s = self.initialize_adam(parameters)
        logging.info("v[\"dW1\"] = " + str(v["dW1"]))
        logging.info("v[\"db1\"] = " + str(v["db1"]))
        logging.info("v[\"dW2\"] = " + str(v["dW2"]))
        logging.info("v[\"db2\"] = " + str(v["db2"]))
        logging.info("s[\"dW1\"] = " + str(s["dW1"]))
        logging.info("s[\"db1\"] = " + str(s["db1"]))
        logging.info("s[\"dW2\"] = " + str(s["dW2"]))
        logging.info("s[\"db2\"] = " + str(s["db2"]))

    def update_parameters_with_adam(self, parameters, grads, v, s, t, learning_rate = 0.01,
                                    beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        """
        Update parameters using Adam
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """
        
        L = len(parameters) // 2                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary
        
        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            ### START CODE HERE ### (approx. 2 lines)
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
            ### END CODE HERE ###

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            ### START CODE HERE ### (approx. 2 lines)
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1 ** t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1 ** t)
            ### END CODE HERE ###

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            ### START CODE HERE ### (approx. 2 lines)
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (grads["dW" + str(l+1)] ** 2)
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (grads["db" + str(l+1)] ** 2)
            ### END CODE HERE ###

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            ### START CODE HERE ### (approx. 2 lines)
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2 ** t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2 ** t)
            ### END CODE HERE ###

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            ### START CODE HERE ### (approx. 2 lines)
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
            ### END CODE HERE ###

        return parameters, v, s

    def update_parameters_with_adam_test_case(self):
        np.random.seed(1)
        v, s = ({'dW1': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
            [ 0.]]), 'db2': np.array([[ 0.],
            [ 0.],
            [ 0.]])}, {'dW1': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
            [ 0.]]), 'db2': np.array([[ 0.],
            [ 0.],
            [ 0.]])})
        W1 = np.random.randn(2,3)
        b1 = np.random.randn(2,1)
        W2 = np.random.randn(3,3)
        b2 = np.random.randn(3,1)

        dW1 = np.random.randn(2,3)
        db1 = np.random.randn(2,1)
        dW2 = np.random.randn(3,3)
        db2 = np.random.randn(3,1)
        
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        
        return parameters, grads, v, s
        
    def tc4(self):
        # 构造参数W1∈2×3, b1∈2×1, W2∈3×3, b2∈3×1
        #  dW1∈2×3, db1∈2×1, dW2∈3×3, db2∈3×1
        # vdW1∈2×3, vdb1∈2×1, vdW2∈3×3, vdb2∈3×1
        # sdW1∈2×3, sdb1∈2×1, sdW2∈3×3, sdb2∈3×1
        parameters, grads, v, s = self.update_parameters_with_adam_test_case()
        # 遍历每一层，执行W = W - α·vdW / ( sqrt(sdW) + ε)
        parameters, v, s  = self.update_parameters_with_adam(parameters, grads, v, s, t = 2)

        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
        print("v[\"dW1\"] = " + str(v["dW1"]))
        print("v[\"db1\"] = " + str(v["db1"]))
        print("v[\"dW2\"] = " + str(v["dW2"]))
        print("v[\"db2\"] = " + str(v["db2"]))
        print("s[\"dW1\"] = " + str(s["dW1"]))
        print("s[\"db1\"] = " + str(s["db1"]))
        print("s[\"dW2\"] = " + str(s["dW2"]))
        print("s[\"db2\"] = " + str(s["db2"]))

    def load_dataset(self):
        np.random.seed(3)
        train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
        # Visualize the data
        # plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        
        return train_X, train_Y

    def tc5(self):
        ''' 数据可视化 '''
        train_X, train_Y = self.load_dataset()
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.flatten(), s=20, cmap=plt.cm.Spectral);
        plt.show()

    def initialize_parameters(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        b1 -- bias vector of shape (layer_dims[l], 1)
                        Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                        bl -- bias vector of shape (1, layer_dims[l])
                        
        Tips:
        - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
        This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
        - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
        """
        
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims) # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
            assert(parameters['W' + str(l)].shape == layer_dims[l], 1)
            
        return parameters

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation (and computes the loss) presented in Figure 2.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape ()
                        b1 -- bias vector of shape ()
                        W2 -- weight matrix of shape ()
                        b2 -- bias vector of shape ()
                        W3 -- weight matrix of shape ()
                        b3 -- bias vector of shape ()
        
        Returns:
        loss -- the loss function (vanilla logistic loss)
        """
        
        # retrieve parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        z1 = np.dot(W1, X) + b1
        a1 = self.relu(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = self.relu(z2)
        z3 = np.dot(W3, a2) + b3
        a3 = self.sigmoid(z3)
        
        cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
        
        return a3, cache

    def backward_propagation(self, X, Y, cache):
        """
        Implement the backward propagation presented in figure 2.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
        cache -- cache output from forward_propagation()
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        m = X.shape[1]
        (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
        
        dz3 = 1./m * (a3 - Y)
        dW3 = np.dot(dz3, a2.T)
        db3 = np.sum(dz3, axis=1, keepdims = True)
        
        da2 = np.dot(W3.T, dz3)
        dz2 = np.multiply(da2, np.int64(a2 > 0))
        dW2 = np.dot(dz2, a1.T)
        db2 = np.sum(dz2, axis=1, keepdims = True)
        
        da1 = np.dot(W2.T, dz2)
        dz1 = np.multiply(da1, np.int64(a1 > 0))
        dW1 = np.dot(dz1, X.T)
        db1 = np.sum(dz1, axis=1, keepdims = True)
        
        gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                    "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                    "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
        
        return gradients

    def sigmoid(self, x):
        """
        Compute the sigmoid of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(x)
        """
        s = 1/(1+np.exp(-x))
        return s

    def relu(self, x):
        """
        Compute the relu of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- relu(x)
        """
        s = np.maximum(0,x)
        
        return s

    def compute_cost(self, a3, Y):
        
        """
        Implement the cost function
        
        Arguments:
        a3 -- post-activation, output of forward propagation
        Y -- "true" labels vector, same shape as a3
        
        Returns:
        cost - value of the cost function
        """
        m = Y.shape[1]
        
        logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
        cost = 1./m * np.sum(logprobs)
        
        return cost
    
    def update_parameters_with_gd(self, parameters, grads, learning_rate):
        """
        Update parameters using one step of gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters to be updated:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients to update each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """

        L = len(parameters) // 2 # number of layers in the neural networks

        # Update rule for each parameter
        for l in range(L):
            ### START CODE HERE ### (approx. 2 lines)
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
            ### END CODE HERE ###
            
        return parameters

    def plot_decision_boundary(self, model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)
        # plt.subplots(figsize=(5,4))
        plt.show()
        
    def model(self, X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
        """
        3-layer neural network model which can be run in different optimizer modes.
        
        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        layers_dims -- python list, containing the size of each layer
        learning_rate -- the learning rate, scalar.
        mini_batch_size -- the size of a mini batch
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates 
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates
        num_epochs -- number of epochs
        print_cost -- True to print the cost every 1000 epochs

        Returns:
        parameters -- python dictionary containing your updated parameters 
        """

        L = len(layers_dims)             # number of layers in the neural networks
        costs = []                       # to keep track of the cost
        t = 0                            # initializing the counter required for Adam update
        seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
        
        # 构造各层W、b的默认值
        parameters = self.initialize_parameters(layers_dims)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no initialization required for gradient descent
        elif optimizer == "momentum":
            v = self.initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = self.initialize_adam(parameters)
        
        # Optimization loop
        for i in range(num_epochs):
            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, mini_batch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                a3, caches = self.forward_propagation(minibatch_X, parameters)

                # Compute cost
                cost = self.compute_cost(a3, minibatch_Y)

                # Backward propagation
                grads = self.backward_propagation(minibatch_X, minibatch_Y, caches)

                # Update parameters
                if optimizer == "gd":
                    parameters = self.update_parameters_with_gd(parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    parameters, v = self.update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    parameters, v, s = self.update_parameters_with_adam(parameters, grads, v, s,
                                                                t, learning_rate, beta1, beta2,  epsilon)
            
            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print ("Cost after epoch %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                    
        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        return parameters

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  n-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        p = np.zeros((1,m), dtype = np.int)
        
        # Forward propagation
        a3, caches = self.forward_propagation(X, parameters)
        
        # convert probas to 0/1 predictions
        for i in range(0, a3.shape[1]):
            if a3[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        # print results

        #print ("predictions: " + str(p[0,:]))
        #print ("true labels: " + str(y[0,:]))
        print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
        
        return p

    def predict_dec(self, parameters, X):
        """
        Used for plotting decision boundary.
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (m, K)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        
        # Predict using forward propagation and a classification threshold of 0.5
        a3, cache = self.forward_propagation(X, parameters)
        predictions = (a3 > 0.5)
        return predictions

    def tc6(self):
        train_X, train_Y = self.load_dataset()
        layers_dims = [train_X.shape[0], 5, 2, 1] # 节点个数：300 - 5 - 2 - 1
        parameters = self.model(train_X, train_Y, layers_dims, optimizer = "gd") # 使用普通的梯度下降

        # Predict
        predictions = self.predict(train_X, train_Y, parameters)

        # Plot decision boundary
        plt.title("Model with Gradient Descent optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

    def tc7(self):
        train_X, train_Y = self.load_dataset()
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters = self.model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

        # Predict
        predictions = self.predict(train_X, train_Y, parameters)

        # Plot decision boundary
        plt.title("Model with Momentum optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

    def tc8(self):
        train_X, train_Y = self.load_dataset()
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters = self.model(train_X, train_Y, layers_dims, optimizer = "adam")

        # Predict
        predictions = self.predict(train_X, train_Y, parameters)

        # Plot decision boundary
        plt.title("Model with Adam optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])
        self.plot_decision_boundary(lambda x: self.predict_dec(parameters, x.T), train_X, train_Y)

class Coding2_3(CodingWorks):
    def tc1(self):
        ''' 生成不同量级的α∈[0.0001, 1] '''
        n = 4
        np.random.seed()
        r = -1 * n * np.random.rand(n)
        alpha = 10 ** r
        for i in range(n):
            logging.info('%.5f' % alpha[i])

    def tc2(self):
        ''' 生成不同量级的β∈[0.9, 0.999] '''
        np.random.seed()
        r = 1 + 2 * np.random.rand(4)
        logging.info('1-β：')
        for i in range(4):
            logging.info('%.5f' % (10 ** (-r[i])))
        beta = 1 - 10 **(-r)
        logging.info('β：')
        for i in range(4):
            logging.info('%.5f' % beta[i])
        
    def tc3(self):
        y_hat = tf.constant(36, name='y_hat')      # 定义常量 ŷ=36
        y = tf.constant(39, name='y')              # 定义常量 y=39

        loss = tf.Variable((y - y_hat)**2, name='loss')  # 定义变量，损失函数L=(y-ŷ)^2

        init = tf.global_variables_initializer()   # 初始化Tensors，此时还没有开始执行任何运算
        with tf.Session() as session:              # 创建Session
            session.run(init)                      # 初始化变量
            print(session.run(loss))               # 运行损失函数，并打印

    def tc4(self):
        a = tf.constant(2)
        logging.info(a)

    def tc5(self):
        # w = tf.Variable(0., dtype=tf.float32)
        # cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25.)
        # train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        x = tf.Variable(3, dtype=tf.float32)
        f = tf.Variable(x**2, name='function')
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            # print(session.run(train))

    def tc6(self):
        x = tf.placeholder(tf.int64, name='x')  # 定义占位符
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            logging.info(session.run(2*x, feed_dict={x:3})) # 用的时候再赋值

    def linear_function(self):
        ''' 实现线性函数：W·X+b '''
        np.random.seed(1)
        
        X = tf.constant(np.random.randn(3,1), name = "X")
        W = tf.constant(np.random.randn(4,3), name = "W")
        b = tf.constant(np.random.randn(4,1), name = "b")
        Y = tf.constant(np.random.randn(4,1), name = "Y")
        
        with tf.Session() as sess:
            result = sess.run(tf.add(tf.matmul(W,X), b))
        
        return result

    def tc7(self):
        logging.info( "result = " + str(self.linear_function()))

    def sigmoid_placeholder(self, z):
        ''' 使用tf版的sigmoid函数 '''
        x = tf.placeholder(tf.float32, name="x")
        sigmoid = tf.sigmoid(x)

        with tf.Session() as sess:
            result = sess.run(sigmoid, feed_dict={x:z})
                
        return result

    def sigmoid_variable(self, z):
        x = tf.Variable(float(z), tf.float32)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(tf.sigmoid(x))
        return result

    def tc8(self):
        logging.info ("sigmoid(0) = " + str(self.sigmoid(0)))
        logging.info ("sigmoid(12) = " + str(self.sigmoid(12)))

        logging.info('sigmoid(12) = ' + str(self.sigmoid_variable(0)))
        logging.info('sigmoid(12) = ' + str(self.sigmoid_variable(12)))

    def cost(self, logits, labels):
        '''
        计算以sigmoid未激活函数的成本函数值

        Arguments:
        logits -- 最后一层的z
        labels -- 标签向量y (1 or 0) 
        
        Returns:
        cost -- 计算成本函数值
        '''
        
        z = tf.placeholder(tf.float32, name = "z")
        y = tf.placeholder(tf.float32, name = "y")
        
        # 根据最后一层z和y，计算成本函数结果
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
        
        with tf.Session() as sess:
            cost = sess.run(cost, feed_dict = {z:logits, y:labels})
        
        return cost

    def tc9(self):
        logits = self.sigmoid_placeholder(np.array([0.2,0.4,0.7,0.9]))
        cost = self.cost(logits, np.array([0,0,1,1]))
        logging.info ("cost = " + str(cost))

    def one_hot_matrix(self, labels, C):
        """
        将标注向量labels中的每个数字x转成第x个元素为1其余为0的列向量的集合
                        
        Arguments:
        labels -- 标注向量
        C -- 分类的个数
        
        Returns: 
        one_hot -- one hot matrix
        """
        C = tf.constant(value = C, name = "C")
        
        # 完成转换
        one_hot_matrix = tf.one_hot(labels, C, axis = 0)
        
        with tf.Session() as sess:
            one_hot = sess.run(one_hot_matrix,)
        
        return one_hot
    
    def tc10(self):
        labels = np.array([1,2,3,0,2,1])
        one_hot = self.one_hot_matrix(labels, C = 4)
        print ("one_hot = " + str(one_hot))

    def ones(self, shape):
        """
        构造形状为shape，内容全1的矩阵

        Arguments:
        shape -- 指定形状
            
        Returns: 
        ones -- 返回构造的矩阵
        """

        ones = tf.ones(shape)
        
        with tf.Session() as sess:
            ones = sess.run(ones)
        
        return ones

    def tc11(self):
        logging.info('ones = ' + str(self.ones([3, 2])))


    def load_dataset(self):
        trainDataPath = os.path.join(self.rootDir, 'coding2_3/datasets/train_signs.h5')
        testDataPath = os.path.join(self.rootDir, 'coding2_3/datasets/test_signs.h5')

        train_dataset = h5py.File(trainDataPath, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File(testDataPath, "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def convert_to_one_hot(self, Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y

    def create_placeholders(self, n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.
        
        Arguments:
        n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
        n_y -- scalar, number of classes (from 0 to 5, so -> 6)
        
        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
        
        Tips:
        - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
        In fact, the number of examples during test/train is different.
        """

        ### START CODE HERE ### (approx. 2 lines)
        X = tf.placeholder(tf.float32, shape = [n_x, None], name = "X")
        Y = tf.placeholder(tf.float32, shape = [n_y, None], name = "Y")
        ### END CODE HERE ###
        
        return X, Y

    def tc12(self):
        X, Y = self.create_placeholders(12288, 6)
        logging.info ("X = " + str(X))
        logging.info ("Y = " + str(Y))

    def initialize_parameters(self):
        """
        初始化W、b参数：  W1 : [25, 12288]    b1 : [25, 1] 
                        W2 : [12, 25]       b2 : [12, 1]
                        W3 : [6, 12]        b3 : [6, 1]
        
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
        
        tf.set_random_seed(1)                   # so that your "random" numbers match ours
            
        ### START CODE HERE ### (approx. 6 lines of code)
        W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
        W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
        ### END CODE HERE ###

        parameters = {"W1": W1, "b1": b1,
                    "W2": W2,   "b2": b2,
                    "W3": W3,   "b3": b3}
        
        return parameters

    def tc13(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            parameters = self.initialize_parameters()
            logging.info("W1 = " + str(parameters["W1"]))
            logging.info("b1 = " + str(parameters["b1"]))
            logging.info("W2 = " + str(parameters["W2"]))
            logging.info("b2 = " + str(parameters["b2"]))

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
        
        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                    the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        
        # Retrieve the parameters from the dictionary "parameters" 
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']
        
        ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
        Z1 = tf.add(tf.matmul(W1,X), b1)                       # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2,A1), b2)                      # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3,A2), b3)                      # Z3 = np.dot(W3,Z2) + b3
        ### END CODE HERE ###
        
        return Z3

    def tc14(self):
        tf.reset_default_graph()

        with tf.Session() as sess:
            X, Y = self.create_placeholders(12288, 6)
            parameters = self.initialize_parameters()
            Z3 = self.forward_propagation(X, parameters)
            print("Z3 = " + str(Z3))

    def compute_cost(self, Z3, Y):
        """
        计算成本函数
        
        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
        
        Returns:
        cost - Tensor of the cost function
        """
        
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        
        # 直接计算各损失函数的均值，即成本函数
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        
        return cost

    def tc15(self):
        tf.reset_default_graph()

        with tf.Session() as sess:
            X, Y = self.create_placeholders(12288, 6)
            parameters = self.initialize_parameters()
            Z3 = self.forward_propagation(X, parameters)
            cost = self.compute_cost(Z3, Y)
            print("cost = " + str(cost))


    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        m = X.shape[1]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)
        
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    def model(self, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
            num_epochs = 1500, minibatch_size = 32, print_cost = True):
        """
        Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
        
        Arguments:
        X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
        Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
        X_test -- training set, of shape (input size = 12288, number of training examples = 120)
        Y_test -- test set, of shape (output size = 6, number of test examples = 120)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep consistent results
        seed = 3                                          # to keep consistent results
        (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]                            # n_y : output size
        costs = []                                        # To keep track of the cost
        
        # Create Placeholders of shape (n_x, n_y)
        ### START CODE HERE ### (1 line)
        X, Y = self.create_placeholders(n_x, n_y)
        ### END CODE HERE ###

        # Initialize parameters
        ### START CODE HERE ### (1 line)
        parameters = self.initialize_parameters()
        ### END CODE HERE ###
        
        # Forward propagation: Build the forward propagation in the tensorflow graph
        ### START CODE HERE ### (1 line)
        Z3 = self.forward_propagation(X, parameters)
        ### END CODE HERE ###
        
        # Cost function: Add cost function to tensorflow graph
        ### START CODE HERE ### (1 line)
        cost = self.compute_cost(Z3, Y)
        ### END CODE HERE ###
        
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        ### START CODE HERE ### (1 line)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        ### END CODE HERE ###
        
        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
            
            # Run the initialization
            sess.run(init)
            
            # Do the training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    ### START CODE HERE ### (1 line)
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    ### END CODE HERE ###
                    
                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    logging.info ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)
                    
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            logging.info ("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            logging.info ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            logging.info ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            
            return parameters

    def tcMain(self):
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = self.load_dataset()
        index = 70
        plt.imshow(X_train_orig[index])
        print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
        # plt.show()

        # 扁平化，训练集：12288 × 1080，测试集：12288 × 120，训练标签：6 × 1080，测试标签：6 × 120
        X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
        X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
        # 归一化到[0, 1]
        X_train = X_train_flatten/255.
        X_test = X_test_flatten/255.
        # 转成6行向量的列集合
        Y_train = self.convert_to_one_hot(Y_train_orig, 6)
        Y_test = self.convert_to_one_hot(Y_test_orig, 6)

        print ("number of training examples = " + str(X_train.shape[1]))
        print ("number of test examples = " + str(X_test.shape[1]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_test shape: " + str(X_test.shape))
        print ("Y_test shape: " + str(Y_test.shape))

        parameters = self.model(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()

    # cmd: python -m unittest samples.Chapter4.case4
