#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: histogram_backprojection.py
@Abstract: 
@Time: 2021/04/03 17:07:51
@Requirements: 
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load ROI
roi = cv2.imread(r'C:\Users\wangzy\Desktop\EE7403\bp3.png')
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# Load Target Image
target = cv2.imread(r'C:\Users\wangzy\Desktop\EE7403\bp2.jpg')
target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
# Calculate the Histogram of ROI
roi_hist = cv2.calcHist([roi_hsv], [0,1], None, [180,256], [0,180,0,256])
# Normalize the Histogram which is Used for Projection
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([target_hsv], [0,1], roi_hist, [0,180,0,256], 1)
# Convolution Operating
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
cv2.filter2D(dst, -1, disc, dst)
# Threshold and Binary Summation by Bit
_, threshold = cv2.threshold(dst, 50, 255, 0)
threshold = cv2.merge((threshold, threshold, threshold))
res = cv2.bitwise_and(target, threshold)
# Print the Result
cv2.imshow('Result', res)
cv2.imshow('Target', target)
cv2.imshow('Threshold',threshold)
cv2.waitKey()