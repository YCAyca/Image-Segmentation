# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 19:50:11 2021

@author: aktas

show_components function was taken from https://iq.opengenus.org/connected-component-labeling/

"""



import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_components(labels):
     # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
        
    #Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()

img = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] 
img2 = cv2.imread("peppers.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)[1] 
img3 = cv2.imread("opencv2.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.threshold(img3, 127, 255, cv2.THRESH_BINARY)[1] 

""" 8 connectivity connected components """
num_labels, labels_im = cv2.connectedComponents(img, connectivity=8)
num_labels2, labels_im2 = cv2.connectedComponents(img2, connectivity=8)
num_labels3, labels_im3 = cv2.connectedComponents(img3, connectivity=8)

print(num_labels)
print(num_labels2)
print(num_labels3)


show_components(labels_im)
show_components(labels_im2)
show_components(labels_im3)

""" 4 connectivity connected components """
num_labels, labels_im = cv2.connectedComponents(img, connectivity=4)
num_labels2, labels_im2 = cv2.connectedComponents(img2, connectivity=4)
num_labels3, labels_im3 = cv2.connectedComponents(img3, connectivity=4)

print(num_labels)
print(num_labels2)
print(num_labels3)


show_components(labels_im)
show_components(labels_im2)
show_components(labels_im3)

