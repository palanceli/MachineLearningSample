
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
	def waitToClose(self, img):
		while True:
			cv2.imshow('image', img)
			if cv2.waitKey(20) & 0xFF == 27:
				break
		
		cv2.destroyAllWindows()

	def case01(self):
		# 根据若干接点生成三次样条插值，并绘制
		img = numpy.zeros((350, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		pts = numpy.array([(240 , 200), (220, 320), (120, 200), (20, 320),(0, 160), (20, 0), (120, 120), (220, 0), (240, 120),], numpy.int32)
		pts = pts.reshape((-1, 2)) 	# 变形成一个坐标序列

		x = pts[:, 0]
		y = pts[:, 1]

		tck, u = scipy.interpolate.splprep([x,y], k=3, s=0)
		xInterp = numpy.linspace(0, 1, num=100, endpoint=True)
		out = scipy.interpolate.splev(xInterp, tck)		# 生成插值

		ptsInterp = numpy.zeros((len(xInterp), 2), numpy.float32)
		ptsInterp = ptsInterp.reshape((-1, 2))

		ptsInterp[:, 0] = out[0]
		ptsInterp[:, 1] = out[1]
		cv2.polylines(img, numpy.int32([ptsInterp]), True, (255, 0, 0), 1)	# 多边形
		cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 0, 0), 2)		# 节点

		logging.debug(pts)
		logging.debug(numpy.int32(ptsInterp))
		self.waitToClose(img)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    s = samples()
    s.case02()
