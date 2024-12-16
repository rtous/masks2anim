import cv2
import numpy as np
import os
import traceback
import sys
import masks2anim

if __name__ == "__main__":
    img_masks_opencv = cv2.imread("input_masks.png")
    img_real_opencv = cv2.imread("input_image.jpg")
    
    #masks2anim.lowpoly_image(
    #    img_masks_opencv = image with masks and transparent background
    #    img_real_opencv = original frame (only if need to detect face)
    #    addFace = detect face (True or False)
    #    shadowSize = rimlight size
    #)s
    result = masks2anim.masks2anim(img_masks_opencv, img_real_opencv, True, 5)
    
    cv2.imwrite("output.png", result)    




