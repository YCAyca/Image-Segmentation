# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:21:50 2021

@author: aktas
"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import cv2



""" Plot the data """

data = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
data = np.array([[1, 5], [3, 1], [10, 3], [10, 2], [10, 1], [1, 0], [2, 15], [0.5, 4.9], [5, 3], [7, 13], [18, 18], [1.9, 0.5]]) 
data = np.random.randint(100, size=(2,2))


centers = [[1, 1], [-1, -1], [1, -1]]
data, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# Plot the data
plt.scatter(data[:,0],data[:,1])
plt.xlabel('x'),plt.ylabel('y')
plt.show()

""" Visualize K means for each iteration """

""" create an empty list for each cluster, k is the cluster number """

k = 2
clusters = [[[0 for _ in range(2)] for _ in range(1)] for _ in range(k)]  


for i in range(k):
    clusters[i].pop() #if we dont do that, additional [0,0] points will be stayed added in our data
    
""" Visualize each iteration. """

for i in range(1,10):
    kmeans = KMeans(n_clusters=k, random_state = 0, max_iter=i).fit(data)
    for index,data_point in enumerate(data):
        clusters[kmeans.labels_[index]].append(list(data_point))
        
    for i in range(k):
        clusters[i] = np.array(clusters[i])
        plt.scatter(clusters[i][:,0],clusters[i][:,1])
        clusters[i] = clusters[i].tolist() 
    plt.show()
    

""" 
If you dont want to observe each iteration you can use it without for loop like below, 
clustering will be end when the model converges
not according to a given iteration number :
    
    kmeans = KMeans(n_clusters=k, random_state = 0).fit(data)    
  
    
"""

""" Image Segmentation """

img = cv2.imread("bird1.jpg", cv2.IMREAD_UNCHANGED) 
img = cv2.imread("birds2.jpg", cv2.IMREAD_UNCHANGED)  
img = cv2.imread("peppers3.jpg", cv2.IMREAD_UNCHANGED)  
vectorized = img.reshape((-1,3))


kmeans = KMeans(n_clusters=5, random_state = 0, n_init=5).fit(vectorized)

centers = np.uint8(kmeans.cluster_centers_)
segmented_data = centers[kmeans.labels_.flatten()]
 
segmented_image = segmented_data.reshape((img.shape))
plt.imshow(segmented_image)
plt.pause(1)

  



