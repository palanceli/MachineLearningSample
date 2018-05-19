
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
        ''' 构造数据 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])

        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        logging.info(df)

    def tc2(self):
        ''' 从文件中加载数据 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])

        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        df.to_csv('temp.csv', sep=',', index=0) # 不保留索引列
        df2 = pd.read_csv('temp.csv', sep=',')
        logging.info(df2)

    def tc3(self):
        ''' 统计数据 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])

        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        logging.info(df.describe())
        logging.info(df.head())

    def tc4(self):
        ''' 画出直方图 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])

        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        df.hist()
        plt.show()

    def tc5(self):
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

    def tc6(self):
        ''' 设置describe()分位数 '''
        data = np.array(np.linspace(1, 20, 20), np.uint32).reshape(10, 2)
        logging.info('\n%s' % data)
        df = pd.DataFrame(data, columns=['a', 'b'])
        logging.info('\n%s' % df.describe(percentiles=[.20, .40, .60, .80]))

    def tc7(self):
        ''' 访问数据 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])
        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        logging.info(df['City name'])
        logging.info(df['City name'][1])
        logging.info(df[0:2])

    def tc8(self):
        ''' 控制数据：整列运算 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])
        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        logging.info(population/1000)   # 将整列除以1000
        logging.info(population.apply(lambda val: val > 1000000)) # 返回该列是否大于100万

    def tc9(self):
        ''' 控制数据：插入列 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])
        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        df['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
        df['Population density'] = df['Population'] / df['Area square miles']
        logging.info(df)

    def tc10(self):
        ''' 索引 '''
        city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
        population = pd.Series([852469, 1015785, 485199, 312342])
        df = pd.DataFrame({ 'City name': city_names, 'Population': population })
        logging.info(df)
        logging.info(df.reindex([2, 1, 0, 3]))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main()
