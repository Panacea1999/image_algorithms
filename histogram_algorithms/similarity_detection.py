#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: similarity_detection.py
@Abstract: 
@Time: 2021/04/03 15:44:57
@Requirements: 
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

'''
Hash: Calculate the Hash Value of an Image
cmpHash: Calculate Hamming Distance of Two Hash Values
cmpGrayChannel: Compare the similarity by Gray Channel
cmpRGBChannel: Compare the similarity by RGB Channels
'''

def Hash(img, method='aHash'):
    # aHash: Average Hash
    # dHash: Difference Hash
    # pHash: Perceptual Hash
    hash_res = ''
    if method == 'aHash':
        aHash_img = cv2.resize(img, (8, 8))
        gray_img = cv2.cvtColor(aHash_img, cv2.COLOR_BGR2GRAY)
        s = 0
        # Calculate Average Gray Level
        for i in range(8):
            for j in range(8):
                s += gray_img[i, j]
        average_gray = s/64
        # Calculate aHash
        for i in range(8):
            for j in range(8):
                if gray_img[i, j] > average_gray:
                    hash_res += '1'
                else:
                    hash_res += '0'
        return hash_res
    
    elif method == 'dHash':
        dHash_img = cv2.resize(img, (9, 8))
        gray_img = cv2.cvtColor(dHash_img, cv2.COLOR_BGR2GRAY)
        # Calculate dHash
        for i in range(8):
            for j in range(8):
                if gray_img[i, j] > gray_img[i, j+1]:
                    hash_res = hash_res+'1'
                else:
                    hash_res = hash_res+'0'
        return hash_res
    
    elif method == 'pHash':
        pHash_img = cv2.resize(img, (32, 32))
        gray_img = cv2.cvtColor(pHash_img, cv2.COLOR_BGR2GRAY)
        # Calculate pHash
        dct = cv2.dct(np.float32(gray_img))
        dct_roi = dct[0:8, 0:8]
        hash = []
        avreage = np.mean(dct_roi)
        for i in range(dct_roi.shape[0]):
            for j in range(dct_roi.shape[1]):
                if dct_roi[i, j] > avreage:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash
    else: print('Wrong Input Hash Method')

def cmpHash(hash1, hash2):
    # Compare Hash Value with Hamming Distance
    res = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            res += 1
    return res

def cmpGrayChannel(image1, image2):
    # Calculate Hist in Gray Channel
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 255])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 255])
    # Calculate the Overlap
    res = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            res += (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            res += 1
    res = res / len(hist1)
    return res
 
def cmpRGBChannel(image1, image2, size=(256, 256)):
    # Calculate RGB Hist
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    res = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        res += cmpGrayChannel(im1, im2)
    res = res / 3
    return res

def similarity_detection(path1, path2):
    # Load the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # aHash
    hash1 = Hash(img1)
    hash2 = Hash(img2)
    n1 = cmpHash(hash1, hash2)
    # dHash
    hash1 = Hash(img1, 'dHash')
    hash2 = Hash(img2, 'dHash')
    n2 = cmpHash(hash1, hash2)
    # pHash
    hash1 = Hash(img1, 'pHash')
    hash2 = Hash(img2, 'pHash')
    n3 = cmpHash(hash1, hash2)
    # Compare by RGB Hist
    n4 = cmpRGBChannel(img1, img2)
    # Compare by Singel Channel Hist
    n5 = cmpGrayChannel(img1, img2)
 
    plt.subplot(121)
    plt.imshow(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
    plt.subplot(122)
    plt.imshow(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)))
    plt.suptitle('Similarity Detected by RGB Histograms: ' + str(n4[0]) + '\n'+'Similarity Detected by Gray Histogram: ' + str(n5[0]) + '\n' + 
    'Similarity Detected by aHash: ' + str(1-n1/64) + '\n' + 'Similarity Detected by dHash: ' + str(1-n2/64) + '\n' + 
    'Similarity Detected by pHash: ' + str(1-n3/64) + '\n'
    )
    plt.show()
 
if __name__ == "__main__":
    path1 = r"C:\Users\wangzy\Desktop\EE7403\pepsi1.jpg"
    path2 = r"C:\Users\wangzy\Desktop\EE7403\pepsi2.jpg"
    similarity_detection(path1, path2)