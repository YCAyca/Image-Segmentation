# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:54:25 2021

@author: aktas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:14:29 2021

@author: aktas
"""

import cv2
import numpy as np


""" multiple seed region groowing image segmentation. Up to 5 seed can be selected """


def search_new_seed(classified_pixels, seed_x,seed_y,img, region_mean,threshold):
    img_h, img_w = img.shape
    for i in range(seed_x, img_w):
        for k in range(seed_y, img_h):
            if [i,k] in classified_pixels:
                continue
            if img[i][k] <= region_mean + threshold: #pick new seed
           #     print("found", i,k)
                classified_pixels.append([i,k])
                return [i,k]
 #   print("no neww seed")       
    return [-1,-1]        

def region_growing(img, seed, threshold, connectivity):
    img_height, img_width = img.shape
    segmented_image = img.copy()    
    
    segmented_image = np.zeros((img_width,img_height), dtype=img.dtype)
    segmented_image = [[element + 255 for element in sub_im] for sub_im in segmented_image]
    
  #  print(segmented_image)
    
    seed_quantity = len(seed)
    flags = [1] * seed_quantity # these flags will hold the info about "whether the seed continue growing or it stopped"
    region_means = [0] * seed_quantity # region mean for each region (different region for each seed) 
    region_sizes = [1] * seed_quantity # region size for each region (different region for each seed)  
    label_list = [0] * seed_quantity # different label for each seed this list will be updated before growing starts
    classified_pixels = [] # we shouldn't visite a neighbor pixel already classified in a region
    visited_pixels_intensities_dif = [] # it wil keep intensity values of each neighbor pixel. After "minimmum" comparison, it well be cleared for the next tour
   
    
    print("seed quantity",seed_quantity)
    print("seeds",seed)    
    print("flags",flags)
    print("im w im h")
    print(img_width, img_height)  
    
    """ check if connectivity parameter is compatible"""
    
    if connectivity == 4:
        neighbor_pixels = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbor_pixels = [(-1, -1),(-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1),(0, -1), (0, 1)]
    else:
        raise ValueError("Invalid connectivity choice! put 4 for 4-connectivity and 8 for 8-connectivity")

    
    """ check if seed parameter is compatible"""
    
    label = 0
    for i in range(seed_quantity):
        if seed[i][0] < img_width  and seed[i][1] < img_height: # if the seed is compatible, initialize the necesseary parts for that seed
            classified_pixels.append(seed[i].copy()) # we should keep the track list about the pixels to see if they are already visited or not
            label_list[i] = label
            segmented_image[seed[i][0]][seed[i][1]] = label #each seed has a unique label so we can keep the different regions via this labels
            label += 20 # change the label for next seed
            region_means[i] = img[seed[i][1]][seed[i][0]]  #initialize region means with only seed intensities, 
            pass
        else:
             raise ValueError("Invalid seed selection, seed coordinates can't be out of image borders")

    
    while len(classified_pixels) < img_height * img_width and 1 in flags:
        for k in range(seed_quantity):
            if flags[k] == 0:
                continue
            for i in range(connectivity):
                tmpx_visited = seed[k][0] + neighbor_pixels[i][0]
                tmpy_visited = seed[k][1] + neighbor_pixels[i][1]
                         
                if [tmpx_visited,tmpy_visited] in classified_pixels:
                    continue
                
                if tmpx_visited < img_width and tmpx_visited >= 0 and tmpy_visited < img_height and tmpy_visited >= 0:
                    x_visited = tmpx_visited
                    y_visited = tmpy_visited
                    classified_pixels.append([seed[k][0] + neighbor_pixels[i][0],seed[k][1] + neighbor_pixels[i][1]])
                    if abs(img[y_visited][x_visited] - region_means[k]) <= threshold:
                         segmented_image[seed[k][1] + neighbor_pixels[i][1]][seed[k][0] + neighbor_pixels[i][0]]  = label_list[k]
                         region_means[k] = (region_sizes[k] * region_means[k] + img[y_visited][x_visited] ) / (region_sizes[k] + 1)
                         region_sizes[k] += 1
                         last_x_accepted = seed[k][0] + neighbor_pixels[i][0]
                         last_y_accepted = seed[k][1] + neighbor_pixels[i][1]
                                            
            if seed[k][0] == last_x_accepted and seed[k][1] == last_y_accepted:
                    new_seed = search_new_seed(classified_pixels,last_x_accepted, last_y_accepted,img, region_means[k],threshold)
                    if new_seed == [-1,-1]:
                        flags[k] = 0 #that seed cant grow anymore
                    else:
                        seed[k][0] = new_seed[0]
                        seed[k][1] = new_seed[1]
            else:            
                seed[k][0] = last_x_accepted
                seed[k][1] = last_y_accepted
        
          
  #       print(segmented_image)
  #       print("last seed was:", seed[k][0], seed[k][1])   
    return segmented_image
        
 
img = np.array([[1, 10, 10, 20, 50, 10, 50, 10],
                [50, 1, 1, 1, 1, 1, 1, 10],
                [20, 1, 1, 1, 1, 1, 1, 40],
                [30, 1, 1, 0, 0, 1, 1, 50],
                [40, 1, 1, 0, 0, 1, 1, 20],
                [100, 1, 1, 1, 1, 1, 1, 30],
                [10, 1, 1, 1, 1, 1, 1, 10],
                [10, 100, 250, 250, 220, 200, 20, 10]])    
print(img)
output = region_growing(img, [[1,1],[7,7]] ,5, 4)
print(output)

img2 = cv2.imread("bi2.jpg", 0)   
img2 = cv2.resize(img2, (125,125))


output = region_growing(img2, [[60,60]] ,10, 8)
cv2.imshow("Segmented  Image", np.array(output, dtype=img2.dtype))
cv2.waitKey(0)
cv2.destroyAllWindows() 


        
        
    