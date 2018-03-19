
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
        

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()

    # cmd: python -m unittest samples.Chapter4.case4
