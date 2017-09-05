
# -*- coding:utf-8 -*-

import logging
import os
import random
import pandas
import io
import sklearn
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot

class DictReader(object):
	def __init__(self):
		self.path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'dict.txt')
		# logging.debug(self.path)
		self.lineData = self.readData()

	def readData(self):
		# 返回词典中的 [(词串, 拼音串)， *]
		lineData = []
		with open(self.path) as f:
			for line in f:
				line = line.strip('\n').strip('\r')
				hanzi, pinyin = line.split('\t')
				hanzi = hanzi.strip()
				pinyin = pinyin.strip('{').strip('}').strip(',')
				lineData.append((hanzi, pinyin))
		return lineData

	def ReadRandomLines(self, lineNum):
		# 从词典中随机抽取lineNum条数据，返回值为[(词串, 拼音串)， *]
		result = []

		for i in range(lineNum):
			iLine = int(len(self.lineData) * random.random())
			result.append(self.lineData[iLine])
		return result

class DataCreator(object):
	def readRandomLines(self, dictReader):
		# 随机读取cHanziLines条词 和 cPinyinLines条拼音，返回它们对应的字节数
		weight = 0.01 	# 随机读取的词条数占总词条数的比例关系
		cHanziLines = int(len(dictReader.lineData) * random.random() * weight)
		lineData = dictReader.ReadRandomLines(cHanziLines)
		hanziString = ''
		for (hanzi, pinyin) in lineData:
			hanziString += hanzi
			# logging.debug('%d %s' % (len(hanzi), hanzi))
		cbHanzi = len(hanziString)

		pinyinString = ''
		cPinyinLines = int(len(dictReader.lineData) * random.random()* weight)
		lineData = dictReader.ReadRandomLines(cPinyinLines)
		for (hanzi, pinyin) in lineData:
			pinyinString += pinyin
			# logging.debug('%d %s' % (len(pinyin.encode('utf-8')), pinyin))
		cbPinyin = len(pinyinString)

		return (cHanziLines, cPinyinLines, cbHanzi, cbPinyin)

	def CreateSampleForSingleFeature(self, nSample):
		# 产生nSample个样本：[(中文词条数，文件大小), *]
		result = []
		dictReader = DictReader()
		for i in range(nSample):
			cHanziLines, cPinyinLines, cbHanzi, cbPinyin = self.readRandomLines(dictReader)
			result.append((cHanziLines, cbHanzi))
		return result


	def CreateSampleForDoubleFeatures(self, nSample):
		# 产生nSample个样本：[(中文词条数，拼音条数， 文件大小), *]
		result = []
		dictReader = DictReader()
		for i in range(nSample):
			cHanziLines, cPinyinLines, cbHanzi, cbPinyin = self.readRandomLines(dictReader)
			result.append((cHanziLines, cPinyinLines, cbHanzi + cbPinyin))
		return result

def SingleFeatureLearning():
	# 单变量线性回归学习过程
    dc = DataCreator()
    # 生成训练样本
    cSamples = 30		# 训练样本个数
    samples = dc.CreateSampleForSingleFeature(cSamples) 

    csvData = 'lines,bytes\n'
    for s in samples:
    	csvData += '%d,%d\n' % (s[0], s[1])

    # 将训练样本读入dataFrame
    dataFrame = pandas.read_csv(io.StringIO(csvData.decode('utf-8')))
    logging.debug(dataFrame)

    # 建立线性回归模型
    regr = sklearn.linear_model.LinearRegression()

    # 拟合
    regr.fit(dataFrame['lines'].values.reshape(-1, 1), dataFrame['bytes']) # reshape(-1, 1)是什么意思？

    # 生成测试样本
    cSample = 5			# 测试样本个数
    samples = dc.CreateSampleForSingleFeature(cSample)
    csvTestData = 'lines,bytes\n'
    for s in samples:
    	csvTestData += '%d,%d\n' % (s[0], s[1])

    # 将训练样本读入dataFrame
    testDataFrame = pandas.read_csv(io.StringIO(csvTestData.decode('utf-8')))
    print(testDataFrame)

    # 预测10000条词的大小
    logging.debug(regr.predict(10000))

    # 画图
    # 1. 训练样本的点
    matplotlib.pyplot.scatter(dataFrame['lines'], dataFrame['bytes'], color='blue')

    # 2. 测试样本的点
    matplotlib.pyplot.scatter(testDataFrame['lines'], testDataFrame['bytes'], marker='x', color='green')

    # 3. 拟合直线
    matplotlib.pyplot.plot(dataFrame['lines'], regr.predict(dataFrame['lines'].values.reshape(-1, 1)), color='red')

    # 
    matplotlib.pyplot.title('words num - file bytes relationship')
    matplotlib.pyplot.ylabel('file bytes')
    matplotlib.pyplot.xlabel('words num')

    matplotlib.pyplot.xlim(0)
    matplotlib.pyplot.ylim(0)

    matplotlib.pyplot.show()

class DoubleFeatureLearning(object):
	def createSampleDataFrame(self, dataCreator, cSamples):
	    samples = dataCreator.CreateSampleForDoubleFeatures(cSamples) 

	    csvData = 'hanziLines,pinyinLines,bytes\n'
	    for s in samples:
	    	csvData += '%d,%d,%d\n' % (s[0], s[1], s[2])

	    # 将训练样本读入dataFrame
	    dataFrame = pandas.read_csv(io.StringIO(csvData.decode('utf-8')))
	    # logging.debug(dataFrame)
	    return dataFrame

	def Main(self):
		# 二元线性回归学习过程
	    dataCreator = DataCreator()
	    # 生成训练样本
	    trainingDataFrame = self.createSampleDataFrame(dataCreator, 30)

	    # 建立线性回归模型
	    regr = sklearn.linear_model.LinearRegression()

	    # 拟合
	    regr.fit(trainingDataFrame[['hanziLines', 'pinyinLines']].values.reshape(-1, 2), trainingDataFrame['bytes'])

	    # 生成测试样本
	    testingDataFrame = self.createSampleDataFrame(dataCreator, 5)

	    # 验证预测
	    for testingSample in testingDataFrame.values:
	    	hanziLines = testingSample[0]
	    	pinyinLines = testingSample[1]
	    	bytes = testingSample[2]
	    	predictBytes = regr.predict(testingDataFrame[['hanziLines', 'pinyinLines']].values.reshape(-1, 2))
	    	print('[%d, %d, %d]' % (hanziLines, pinyinLines, bytes))
	    	print(predictBytes)
	    # logging.debug(regr.predict(10000))

	    # # 画图
	    # # 1. 训练样本的点
	    # matplotlib.pyplot.scatter(dataFrame['lines'], dataFrame['bytes'], color='blue')

	    # # 2. 测试样本的点
	    # matplotlib.pyplot.scatter(testDataFrame['lines'], testDataFrame['bytes'], marker='x', color='green')

	    # # 3. 拟合直线
	    # matplotlib.pyplot.plot(dataFrame['lines'], regr.predict(dataFrame['lines'].values.reshape(-1, 1)), color='red')

	    # # 
	    # matplotlib.pyplot.title('words num - file bytes relationship')
	    # matplotlib.pyplot.ylabel('file bytes')
	    # matplotlib.pyplot.xlabel('words num')

	    # matplotlib.pyplot.xlim(0)
	    # matplotlib.pyplot.ylim(0)

	    matplotlib.pyplot.show()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    
    learner = DoubleFeatureLearning()
    learner.Main()
