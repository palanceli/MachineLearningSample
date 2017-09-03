
# -*- coding:utf-8 -*-

import logging
import os
import random

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
		cHanziLines = int(len(dictReader.lineData) * random.random())
		lineData = dictReader.ReadRandomLines(cHanziLines)
		hanziString = ''
		for (hanzi, pinyin) in lineData:
			hanziString += hanzi
			# logging.debug('%d %s' % (len(hanzi.encode('utf-8')), hanzi))
		cbHanzi = len(hanziString.encode('utf-8'))

		pinyinString = ''
		cPinyinLines = int(len(dictReader.lineData) * random.random())
		lineData = dictReader.ReadRandomLines(cPinyinLines)
		for (hanzi, pinyin) in lineData:
			pinyinString += pinyin
			# logging.debug('%d %s' % (len(pinyin.encode('utf-8')), pinyin))
		cbPinyin = len(pinyinString.encode('utf-8'))

		return (cHanziLines, cPinyinLines, cbHanzi, cbPinyin)

	def CreateSampleForSingleFeature(self, nSample):
		# 产生nSample个样本：[(中文词条数，文件大小), *]
		result = []
		dictReader = DictReader()
		for i in range(nSample):
			cHanziLines, cPinyinLines, cbHanzi, cbPinyin = self.readRandomLines(dictReader)
			result.append((cHanziLines, cbHanzi))
		return result


	def CreateSampleForDoubleFeature(self, nSample):
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
    samples = dc.CreateSampleForSingleFeature(10)
    logging.debug(' lines    bytes')
    logging.debug('-------  -------')
    for s in samples:
    	logging.debug('%7d %7d' % (s[0], s[1]))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    
    SingleFeatureLearning()
