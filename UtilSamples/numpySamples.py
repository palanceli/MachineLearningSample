
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2
import scipy
import scipy.interpolate
import matplotlib
import matplotlib.pyplot

class samples(object):
	def case01(self):
		logging.debug('生成[1, 8]的等差数列：')
		x = numpy.linspace(1, 8, 8)
		print(x)
		print('\n')

		logging.debug('reshape(2*4) 的矩阵：')
		print(x.reshape((2, 4)))
		print('\n')
		
		logging.debug('reshape(2*2*2) 的矩阵：')
		print(x.reshape((2, 2, 2)))
		print('\n')

		logging.debug('reshape后的数组和原数组共享一段内存，如果原数组内容变化，变形后的一会变：')
		y = x.reshape(2, 4)
		y[1, 1] = 10
		logging.debug('y:')
		print(y)
		logging.debug('x:')
		print(x)
		print('\n')

		logging.debug('reshape的值为-1，会根据数组长度和剩余维度中推断出来：')
		print(x.reshape(-1, 2, 2))
		print('\n')

	def case02(self):
		logging.debug('初始化 2*3*4 全0矩阵：')
		x = numpy.zeros((2, 3, 4), numpy.uint8)	
		print(x)
		print('\n')

		logging.debug('初始化 2*3*4 全1矩阵：')
		x = numpy.ones((2, 3, 4), numpy.uint8)
		print(x)
		print('\n')

		logging.debug('这3种写法是等价的：')
		x = numpy.array([[1, 2], [3, 4], [5, 6]], numpy.uint8)
		y = numpy.linspace(1, 6, 6, dtype=numpy.uint8).reshape(3, 2)
		z = numpy.zeros((3, 2), numpy.uint8)
		z[:, 0] = [1, 3, 5]
		z[:, 1] = [2, 4, 6]

		print(x)
		print(y)
		print(z)

		logging.debug('全部赋值为2：')
		x = numpy.ones((2, 3, 4), numpy.uint8)
		x[:] = 2
		print(x)
		print('\n')

		logging.debug('修改第1维为1（0 based）的所有数据为4：')
		x = numpy.ones((2, 3, 4), numpy.uint8)
		x[1, :, :] = 4
		print(x)
		print('\n')

		logging.debug('修改第2维为1（0 based）的所有数据为4：')
		x = numpy.ones((2, 3, 4), numpy.uint8)
		x[:, 1, :] = 4
		print(x)
		print('\n')

		logging.debug('修改第3维为1（0 based）的所有数据为4：')
		x = numpy.ones((2, 3, 4), numpy.uint8)
		x[:, :, 1] = 4
		print(x)
		print('\n')

		logging.debug('修改第3维为1（0 based）的所有数据为4：')
		x = numpy.ones((2, 3, 4), numpy.uint8)
		x[:, :, 1] = 4
		print(x)
		print('\n')

	def case03(self):
		logging.debug('将每一维的第1列改为2， 3， 4：')
		x = numpy.ones((2, 3, 4), numpy.uint8)
		x[:, :, 1] = [2, 3, 4]
		print(x)
		print('\n')

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    s = samples()
    s.case02()
