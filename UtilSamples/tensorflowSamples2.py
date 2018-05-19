
# -*- coding:utf-8 -*-
# 使用 TensorFlow 的基本步骤
# 学习目标：
# - 学习基本的 TensorFlow 概念
# - 在 TensorFlow 中使用 LinearRegressor 类并基于单个输入特征预测各城市街区的房屋价值中位数
# - 使用均方根误差 (RMSE) 评估模型预测的准确率
# - 通过调整模型的超参数提高模型准确率
#

import unittest
import logging
import os

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

class Samples(unittest.TestCase):
    ''' 数据基于加利福尼亚州 1990 年的人口普查数据 '''
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        tf.logging.set_verbosity(tf.logging.ERROR)  # 训练过程中输出相关信息
        pd.options.display.max_rows = 10            # 最多显示10行
        pd.options.display.float_format = '{:.1f}'.format   # 设置浮点型格式
        
    def loadData(self):
        california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
        # 重新洗牌
        # np.random.permutation(...)返回打乱顺序的数组
        california_housing_dataframe = california_housing_dataframe.reindex(
            np.random.permutation(california_housing_dataframe.index))
        california_housing_dataframe["median_house_value"] /= 1000.0
        return california_housing_dataframe

    def tc1(self):
        ''' 检查数据 '''
        california_housing_dataframe = self.loadData()
        logging.info(california_housing_dataframe.describe())


    def tc2(self):
        ''' 检查数据 '''
        california_housing_dataframe = self.loadData()
        logging.info(california_housing_dataframe["total_rooms"])

    def my_input_fn(self, features, targets, batch_size=1, shuffle=True, num_epochs=None):
        """ 第 4 步：定义输入函数
        训练单特征的线性回归模型
        Trains a linear regression model of one feature.
    
        Args:
        features:   pandas DataFrame of features
        targets:    pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle:    True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
        Returns:
        Tuple of (features, labels) for next data batch
        """
    
        # 首先，将 Pandas 特征数据转换成 NumPy 数组字典。
        features = {key:np.array(value) for key,value in dict(features).items()}                                           
    
        # 构建 Dataset 对象，并配置batch_size 和 num_epochs
        ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)
        
        # Shuffle the data, if specified
        if shuffle:
            ds = ds.shuffle(buffer_size=10000) # 对数据进行随机处理，以便数据在训练期间以随机方式传递到模型
        
        # 构建迭代器，并向返回下一个batch的数据。
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def tcMain(self):
        ''' 将 total_rooms 作为输入特征，尝试预测 median_house_value '''
        california_housing_dataframe = self.loadData()
        # 第 1 步：定义特征并配置特征列
        # Define the input feature: total_rooms.
        my_feature = california_housing_dataframe[["total_rooms"]] # 提取列，双中括号内可以有多个列名

        #   使用“特征列”的结构来表示特征的数据类型。特征列仅存储对特征数据的描述；不包含特征数据本身。
        feature_columns = [tf.feature_column.numeric_column("total_rooms")]
        
        # 第 2 步：定义目标
        targets = california_housing_dataframe["median_house_value"]

        # 第 3 步：配置 LinearRegressor
        #   使用 LinearRegressor 配置线性回归模型，并使用 GradientDescentOptimizer（实现小批量随机梯度下降法 (SGD)）
        #   训练该模型。learning_rate 参数可控制梯度步长的大小。
        #   注意：为了安全起见，通过 clip_gradients_by_norm 将梯度裁剪应用到优化器。确保梯度大小在训练期间不会变得过大，
        #   梯度过大会导致梯度下降法失败。
        my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

        # Configure the linear regression model with our feature columns and optimizer.
        # Set a learning rate of 0.0000001 for Gradient Descent.
        linear_regressor = tf.estimator.LinearRegressor(
            feature_columns=feature_columns,
            optimizer=my_optimizer
        )
        # 第 4 步：见函数my_input_fn(...)

        # 第 5 步：训练模型
        _ = linear_regressor.train(input_fn = lambda:self.my_input_fn(my_feature, targets), steps=100)

        # 第 6 步：评估模型
        # Create an input function for predictions.
        # Note: Since we're making just one prediction for each example, we don't 
        # need to repeat or shuffle the data here.
        prediction_input_fn =lambda: self.my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

        # Call predict() on the linear_regressor to make predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)

        # Format predictions as a NumPy array, so we can calculate error metrics.
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Print Mean Squared Error and Root Mean Squared Error.
        mean_squared_error = metrics.mean_squared_error(predictions, targets)
        root_mean_squared_error = math.sqrt(mean_squared_error)
        logging.info("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
        logging.info("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
        

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
