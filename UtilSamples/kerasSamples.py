
# -*- coding:utf-8 -*-

import unittest
import logging
import os
import numpy as np
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import PIL
import h5py
from matplotlib.pyplot import imshow
import scipy
import scipy.io
import scipy.misc
from PIL import Image
from scipy import ndimage
import sklearn
import sklearn.datasets
import sklearn.linear_model
import tensorflow as tf
from tensorflow.python.framework import ops

from keras import layers
from keras.layers import Input, Lambda, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

import keras.backend as K
from matplotlib.pyplot import imshow
from keras.models import Sequential

class Samples(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
        self.rootDir = os.path.join(os.path.expanduser('~'), 'Documents/DeepLearningAI作业/')
        np.random.seed(1)
        plt.switch_backend('Qt5Agg') # 在独立窗口中弹出绘图，而不是和命令行共用一个窗口
        # plt.subplots(figsize=(5,4))  # 调整窗口大小
        plt.rcParams['figure.figsize'] = (5.0, 4.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        K.set_image_data_format('channels_last')
        K.set_learning_phase(1)

    def tc1(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        # Generate dummy data
        import numpy as np
        data = np.random.random((1000, 100))
        labels = np.random.randint(2, size=(1000, 1))

        # Train the model, iterating on the data in batches of 32 samples
        model.fit(data, labels, epochs=10, batch_size=32)
        # 将happyModel转成图片，需要安装Graphviz:
        # brew install graphviz
        plot_model(model, to_file='model.png')
        SVG(model_to_dot(model).create(prog='dot', format='svg'))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
