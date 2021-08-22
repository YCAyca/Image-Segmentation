# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 20:41:02 2021

@author: aktas
"""

import cv2
import numpy as np

""" Basic Thresholding """

img = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("journal.jpg", cv2.IMREAD_GRAYSCALE)

_, segmented1 = cv2.threshold(img2, 127,255,cv2.THRESH_BINARY)
print(segmented1)

cv2.imshow("Simple Segmented Output Image", segmented1)
cv2.waitKey(0) 
cv2.destroyAllWindows()


_, segmented2 = cv2.threshold(img2, 127,1,cv2.THRESH_BINARY)
segmented2 = segmented2.astype(dtype='f')
print(segmented2)

cv2.imshow("Simple Segmented Output Image", segmented2)
cv2.waitKey(0) 
cv2.destroyAllWindows()

""" Otsu Thresholding """

thresh, segmented3 = cv2.threshold(img2, 127,1,cv2.THRESH_OTSU)
segmented3 = segmented3.astype(dtype='f')
print(thresh)
print(segmented3)

cv2.imshow("Otsu Segmented Output Image", segmented3)
cv2.waitKey(0) 
cv2.destroyAllWindows()

""" Adaptive Thresholding """

segmented4 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
print(segmented4)

cv2.imshow(" Adaptive Mean Segmented Output Image", segmented4)
cv2.waitKey(0) 

segmented5 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
print(segmented5)

cv2.imshow("Adaptive Gaussian Segmented Output Image", segmented5)
cv2.waitKey(0) 

cv2.destroyAllWindows()

