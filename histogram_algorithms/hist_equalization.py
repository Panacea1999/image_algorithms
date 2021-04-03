#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: hist_equalization.py
@Abstract: 
@Time: 2021/04/02 15:49:06
@Requirements: 
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
img = cv2.imread(r'C:\Users\wangzy\Desktop\EE7403\pic2.jpeg', cv2.IMREAD_GRAYSCALE)
# HE
equ = cv2.equalizeHist(img)
# Print Results
cv2.imshow("Originial", img)
cv2.imshow("HE Result", equ)
# Print Hist
plt.figure(figsize=(6,3))
plt.title("Histogram of Original Image")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.hist(img.ravel(), 256, color='darkcyan')

plt.figure(figsize=(6,3))
plt.title("Histogram of Enhanced Image")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.hist(equ.ravel(), 256, color='darkcyan')
plt.show()
 
cv2.waitKey(0)
cv2.destroyAllWindows()