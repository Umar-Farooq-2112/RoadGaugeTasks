import cv2
import numpy as np


def binary_mask(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_img = np.zeros_like(grayscale_image)
    binary_img[grayscale_image!= 0] = 255
    return binary_img
    

def apply_threshold(image,max_threshold):
    different_pixels = []
    for i,item in enumerate(image):
        if np.count_nonzero(item)>0:
            different_pixels.append(np.count_nonzero(item))
        else:
            different_pixels.append(0)
    different_pixels = np.array(different_pixels)
    return np.mean(different_pixels)>max_threshold

def road_edge_detector(image,mask,max_threshold = 50):
        
    image = binary_mask(image)
    mask = binary_mask(mask)

    merged = cv2.bitwise_xor(image,mask)
    
    left_side = merged[:,:int(merged.shape[1]/2)]
    right_side = merged[:,int(merged.shape[1]/2):]

    left =  apply_threshold(left_side,max_threshold)
    right =  apply_threshold(right_side,max_threshold)

    return left,right
