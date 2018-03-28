
# -*- coding:utf-8 -*-

import unittest
import logging
import os
import numpy as np
import scipy
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt

class Samples(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        np.random.seed(1)
        plt.switch_backend('Qt5Agg') # 在独立窗口中弹出绘图，而不是和命令行共用一个窗口

    def tc1(self):
    	data = np.random.randn(100)
    	plt.hist(data, 100)
    	plt.show()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
