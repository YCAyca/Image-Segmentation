# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:43:59 2021

@author: aktas
"""

import cv2
from skimage.feature import canny
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


img = cv2.imread("peppers2.jpg", 0) 
#img = cv2.resize(img, (150,188)) 

"""Edge Based Segmentation """

""" edge detection with canny """

edges = canny(img/225.)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(edges, cmap=plt.cm.gray)
ax.axis('off')
ax.set_title('Canny detector')

""" region - hole filling """


fill_holes = ndi.binary_fill_holes(edges)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_holes, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Filling the holes')

