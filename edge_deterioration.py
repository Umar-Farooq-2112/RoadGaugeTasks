import numpy as np
from edge_detector import binary_mask

def crop_required_image(mask,crop_ratio=0.1):
    for i,item in enumerate(mask):
        if np.count_nonzero(item)!=0:
            break
    for j in range(mask.shape[0]-1,-1,-1):
        if np.count_nonzero(mask[j])!=0:
            break
    mask = mask[i:j,:]
    height = mask.shape[0]
    mask = mask[int(height*crop_ratio):height-int(height*crop_ratio),:]
    return mask

def first_nonzero_indices(arr):
    indices = np.argmax(arr != 0, axis=1)
    all_zero_rows = np.all(arr == 0, axis=1)
    indices[all_zero_rows] = -1
    return indices

def best_fit_line(mask):    
    x = []
    y = []
    for i,item in enumerate(mask):
        for j,subitem in enumerate(item):
            if subitem!=0:
                y.append(j)
                x.append(i)
                break
    
    coefficients = np.polyfit(x, y, 1)
    poly_function = np.poly1d(coefficients)

    return poly_function

def get_difference_values(mask,line_function):
    indices = first_nonzero_indices(mask)
    difference = []
    for i in range(len(mask)):
        if indices[i]!=-1:
            difference.append(abs(indices[i]-line_function(i).astype(int)))
        else:
            difference.append(-1)
    return difference

def detect_deterioration(image,threshold=25):
    mask = binary_mask(image)
    mask = crop_required_image(mask)
    line_function = best_fit_line(mask)
    difference = get_difference_values(mask,line_function)
    return np.max(difference)>threshold
