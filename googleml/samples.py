
# -*- coding:utf-8 -*-

import logging
import unittest

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
# from tensorflow.python.data import Dataset

class samples(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        tf.logging.set_verbosity(tf.logging.ERROR)
        pd.options.display.max_rows = 10
        pd.options.display.float_format = '{:.1f}'.format

class first_stpes_with_tf(samples):
	def case(self):
		california_housing_dataframe = pd.read_csv("./california_housing_train.csv", sep=",")

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    
    unittest.main
    # cmd: python -m unittest samples.Chapter4.case4
