
# -*- coding:utf-8 -*-

import unittest
import logging
import os
import numpy as np
import scipy
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

class Samples(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        np.random.seed(1)
        plt.switch_backend('Qt5Agg') # 在独立窗口中弹出绘图，而不是和命令行共用一个窗口

    def tc1(self):
        ''' 演示describe()返回值 '''
        data = np.array([1, 1, 1, 2, 3, 4, 5, 6], np.uint32)
        df = pd.DataFrame(data, columns=['number'])
        logging.info(data)
        logging.info('\n%s' % df.describe())
        '''
        输出结果如下：
                number
        count  8.00000
        mean   2.87500  平均值
        std    1.95941  标准差 δ=sqrt(1/n sum( (xi - μ)**2 ) )
        min    1.00000  最小数
        25%    1.00000  四分位数，第2个和第3个数的均值
        50%    2.50000  第二四分位数，第4和第5个数的均值
        75%    4.25000  
        max    6.00000  最大数
        '''

    def tc2(self):
        ''' 设置describe()分位数 '''
        data = np.array(np.linspace(1, 20, 20), np.uint32).reshape(10, 2)
        logging.info('\n%s' % data)
        df = pd.DataFrame(data, columns=['a', 'b'])
        logging.info('\n%s' % df.describe(percentiles=[.20, .40, .60, .80]))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
