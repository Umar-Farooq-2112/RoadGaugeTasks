import cv2
from edge_detector import road_edge_detector
from edge_deterioration import detect_deterioration


def edge_deterioration_detector(image,mask,edge_threshold=50,deterioration_threshold = 40):
    left,right = road_edge_detector(image,mask,edge_threshold)
    if left and right:
        left = detect_deterioration(mask,deterioration_threshold)
        right = detect_deterioration(cv2.flip(mask,1),deterioration_threshold)
    elif left:
        left = detect_deterioration(mask,deterioration_threshold)
        right = None
    elif right:
        left = None
        right = detect_deterioration(cv2.flip(mask,1),deterioration_threshold)
    else:
        print("No edge Detected on either side of road")
        return None,None
    
    return left, right
