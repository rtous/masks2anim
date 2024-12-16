import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np
import os
import traceback
import sys
import dlib
from imutils import face_utils

def opencv_to_RGB(c):
    return c[::-1]

def imageFromPath(inputpath):   
    print("faceFromPath("+inputpath+", im_res)")
    im = cv2.imread(inputpath)
    if im is None:
        path, file_extension = os.path.splitext(inputpath)
        if file_extension==".png":
            print("failed, testing with path: "+path+".jpg")
            im = cv2.imread(path+".jpg")
            print("failed, testing with path: "+path+".png")
        elif file_extension==".jpg":
            im = cv2.imread(path+".png")
    assert im is not None, "file could not be read, check with os.path.exists()"
    return im

def simplify(opencvContour, tolerance = 4.0):#5.0 , preserve_topology=False
    """ Simplify a polygon with shapely.
    Polygon: ndarray
        ndarray of the polygon positions of N points with the shape (N,2)
    """
    polygon = np.squeeze(opencvContour)
    poly = shapely.geometry.Polygon(polygon)
    poly_s = poly.simplify(tolerance=tolerance, preserve_topology=False)
    # convert it back to numpy
    coords = np.array(poly_s.boundary.coords[:])
    #Convert shapely polygon (N, 2) to opencv contour (N-1, 1, 2)
    opencvContourSimplified = coords.reshape((-1,1,2)).astype(np.int32)    
    return opencvContourSimplified

def addAlpha(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def pixelate(input, w, h): # w,h  Desired "pixelated" size
    height, width = input.shape[:2]
    # Resize input to "pixelated" size
    temp = cv2.resize(input, (w, h),  interpolation=cv2.INTER_NEAREST)#cv2.INTER_LINEAR antialiasing
    # Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output

def getContours(im):
    height, width = im.shape[:2]
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    contours_not_dilated = []
    contours_raw = []
    contours_simplified = []
    colors = []

    #split image in C color regions (with a minimum of 1000 pixels)
    selected_contours = []
    contours_simplified = [] 
    colorNum = 0
    totalContours = 0
    #unique = np.unique(imgray)
    unique_colours = np.unique(im.reshape(-1, im.shape[2]), axis=0)
    #For each COLOR 
    for i, color in enumerate(unique_colours):
        mask = cv2.inRange(im, color, color)
        area = cv2.countNonZero(mask)
        #if the color covers more than the half of the pixels
        #we asume that it's background
        if area > 200 and area < height*width/2: #avoid the frame contour
            #split color mask in N contours (with a minimum of area > 10)
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10:
                    contours_not_dilated.append(contour)
                    #dilate 1 pixel (to avoid gaps between simplified contours)
                    #remove anything outside the contour
                    part_mask = cropContours(mask, contour)
                    kernel = np.ones((4, 4), np.uint8)
                    part_mask = cv2.dilate(part_mask, kernel, iterations=1)

                    #find contours again
                    ret, thresh = cv2.threshold(part_mask, 127, 255, 0)
                    image, contours_dilated, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    max_contour = max(contours_dilated, key = cv2.contourArea)

                    contours_raw.append(max_contour)
                    colors.append(opencv_to_RGB(color))
                    test = np.zeros_like(imgray)
                    totalContours = totalContours+1                          
            colorNum = colorNum+1
    return contours_not_dilated, contours_raw, colors

def cropContours(im, contour):
    im_res = np.zeros_like(im)
    cv2.fillPoly(im_res, pts =[contour], color=(255,255,255))
    return im_res

def simplifyContours(contours):
    contours_simplified = []
    for contour in contours:
        try:
            simplifiedContour = simplify(contour)
            contours_simplified.append(simplifiedContour)
        except:
            print("Contour discarded as contains multi-part geometries")
            print(traceback.format_exc())
            print("Using original contour without simplification")
            contours_simplified.append(contour)
    return contours_simplified

def fillContours(contours, colors, imcolor):
    for i, contour in enumerate(contours):
        display_color = (int(colors[i][2]),int(colors[i][1]),int(colors[i][0]),255)
        cv2.fillPoly(imcolor, pts =[contour], color=display_color)

    imcolor_pixelated = pixelate(imcolor, 512, 512)
    #imcolor_pixelated = imcolor
    return imcolor_pixelated

def change_brightness(img, value=100):
    _, _, _, a_channel = cv2.split(img)
    #img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = np.dstack((img, a_channel))
    return img

def addShadow(imcolor, shadowSize=10):    
    imcolor_result_shadow = imcolor.copy()
    height, width = imcolor_result_shadow.shape[:2]
    offsetx = shadowSize
    offsety = 0
    M = np.float32([[1, 0, offsetx], [0, 1, offsety]])
    dst_mat = np.zeros((height, width, 4), np.uint8)
    size = (width, height)
    imcolor_result_shadow = cv2.warpAffine(imcolor_result_shadow, M, size, dst_mat)
    imcolor_result_shadow = change_brightness(imcolor_result_shadow)
    result = overlay(bottomImage=imcolor_result_shadow, topImage=imcolor)
    return result

def overlay(bottomImage, topImage):
	#Idea: add the topImage (complete) to a sliced bottomImage 
    #Obtain an opencvmask from the alpha channel of the topImage
    _, mask = cv2.threshold(topImage[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    #Invert the mask
    mask = cv2.bitwise_not(mask) 
    #Use the mask to cut the intersection from the bottomImage
    bottomImageMinusTopImage = cv2.bitwise_and(bottomImage, bottomImage, mask=mask)
    #Add the topImage (complete) and bottomImageMinusTopImage
    result = bottomImageMinusTopImage + topImage
    return result

def drawContours(contours, colors, imcolor):

    for i, contour in enumerate(contours):
        #color_id = idFromColor(palette, colors[i])
        color_id = idFromColor(palette, colors[i])
        if color_id in color_assignmentNEW:
            display_color = color_assignmentNEW[color_id]
        else:
            display_color = random_255_colors_4_channels[i]
        
        if PRINT_COLOR_IDS:
            # compute the center of the contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(imcolor, str(color_id), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0,255), 2)
        cv2.drawContours(imcolor, [contour], contourIdx=0, color=display_color, thickness=1)        
    #imcolor_pixelated = pixelate(imcolor, 512, 512)
    return imcolor

################
# FACE
################

#WARNING: landmark number starts from 0, so substract 1 to the number from the reference photo
def face(im, im_res):
    # initialize built-in face detector in dlib
    detector = dlib.get_frontal_face_detector()
    # initialize face landmark predictor
    PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"#https://github.com/davisking/dlib-models
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    #resize to width 500
    #im_resized = cv2.resize(im, (500, 500), interpolation= cv2.INTER_LINEAR)
    #Resizing accelerates but you need to upscale everything later
    image = im
    #image = imutils.resize(im, width=500)
    #convert it to grayscale
    im_resized_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(im_resized_gray, 1)
    #for each face
    for (i, rect) in enumerate(rects):
        # predict facial landmarks in image and convert to NumPy array
        shape = predictor(im_resized_gray, rect)
        draw_eyes_contour(shape, im_res)
        draw_pupils(shape, im_res) 
        draw_nose(shape, im_res)
        draw_mouth(shape, im_res)

def draw_nose(shape, im_res):
    eye_contour = np.array([
            [shape.part(31).x, shape.part(31).y], [shape.part(33).x, shape.part(33).y], 
            [shape.part(33).x, shape.part(33).y], [shape.part(35).x, shape.part(35).y], 
            [shape.part(35).x, shape.part(35).y], [shape.part(31).x, shape.part(31).y]
            ], np.int32)
    
    pts = eye_contour.reshape((-1, 1, 2))
    #im_res = cv2.polylines(im_res, [pts], True, (199, 199, 199, 255), 1)
    cv2.fillPoly(im_res, [pts], color = (118, 113, 168, 255))

def draw_mouth(shape, im_res):
    #The numbers in the image start in 1, so need to reduce them by 1

    '''
    thickness = 
    eye_contour = np.array([
            [shape.part(48).x, shape.part(48).y], [shape.part(54).x, shape.part(54).y]
            ], np.int32)
    
    pts = eye_contour.reshape((-1, 1, 2))
    im_res = cv2.polylines(im_res, [pts], True, (199, 199, 199, 255), 1)
    '''
    
    x_plus_top = int(shape.part(49).x - shape.part(53).x)/3
    x_plus_bottom = int(shape.part(55).x - shape.part(59).x)/3
    y_plus_left = int(shape.part(49).y - shape.part(59).y)/3
    y_plus_right = int(shape.part(53).y - shape.part(55).y)/3
    '''eye_contour = np.array([
            [shape.part(49).x + x_plus_top, shape.part(49).y - y_plus_left], [shape.part(53).x - x_plus_top, shape.part(53).y - y_plus_right], 
            [shape.part(53).x - x_plus_top, shape.part(53).y - y_plus_right], [shape.part(55).x + x_plus_bottom, shape.part(55).y + y_plus_right], 
            [shape.part(55).x + x_plus_bottom, shape.part(55).y + y_plus_right], [shape.part(59).x - x_plus_bottom, shape.part(59).y + y_plus_right],
            [shape.part(59).x - x_plus_bottom, shape.part(59).y + y_plus_left], [shape.part(49).x + x_plus_top, shape.part(49).y - y_plus_left]
            ], np.int32)'''

    shape = face_utils.shape_to_np(shape)
    
    '''
    lips_contour = shape[48:59] #mouth interior
    pts = lips_contour.reshape((-1, 1, 2))
    #im_res = cv2.polylines(im_res, [pts], True, (199, 199, 199, 255), 1)
    cv2.fillPoly(im_res, [pts], color = (118, 113, 168, 255))
    '''

    mouth_contour = shape[60:67] #mouth interior
    pts = mouth_contour.reshape((-1, 1, 2))
    #im_res = cv2.polylines(im_res, [pts], True, (199, 199, 199, 255), 1)
    cv2.fillPoly(im_res, [pts], color = (118, 113, 168, 255))

def draw_mouthOLD(shape, im_res):
    '''
    thickness = 
    eye_contour = np.array([
            [shape.part(48).x, shape.part(48).y], [shape.part(54).x, shape.part(54).y]
            ], np.int32)
    
    pts = eye_contour.reshape((-1, 1, 2))
    im_res = cv2.polylines(im_res, [pts], True, (199, 199, 199, 255), 1)
    '''

    
    x_plus_top = int(shape.part(49).x - shape.part(53).x)/3
    x_plus_bottom = int(shape.part(55).x - shape.part(59).x)/3
    y_plus_left = int(shape.part(49).y - shape.part(59).y)/3
    y_plus_right = int(shape.part(53).y - shape.part(55).y)/3
    eye_contour = np.array([
            [shape.part(49).x + x_plus_top, shape.part(49).y - y_plus_left], [shape.part(53).x - x_plus_top, shape.part(53).y - y_plus_right], 
            [shape.part(53).x - x_plus_top, shape.part(53).y - y_plus_right], [shape.part(55).x + x_plus_bottom, shape.part(55).y + y_plus_right], 
            [shape.part(55).x + x_plus_bottom, shape.part(55).y + y_plus_right], [shape.part(59).x - x_plus_bottom, shape.part(59).y + y_plus_right],
            [shape.part(59).x - x_plus_bottom, shape.part(59).y + y_plus_left], [shape.part(49).x + x_plus_top, shape.part(49).y - y_plus_left]
            ], np.int32)
    
    pts = eye_contour.reshape((-1, 1, 2))
    #im_res = cv2.polylines(im_res, [pts], True, (199, 199, 199, 255), 1)
    cv2.fillPoly(im_res, [pts], color = (118, 113, 168, 255))
    

def draw_eyes_contour(shape, im_res):
    x_plus_top = int(shape.part(37).x - shape.part(44).x)/4
    x_plus_bottom = int(shape.part(46).x - shape.part(41).x)/4
    y_plus_left = int(shape.part(37).y - shape.part(41).y)/5
    y_plus_right = int(shape.part(44).y - shape.part(46).y)/5
    eye_contour = np.array([
            [shape.part(37).x + x_plus_top, shape.part(37).y + y_plus_left], [shape.part(44).x - x_plus_top, shape.part(44).y + y_plus_right], 
            [shape.part(44).x - x_plus_top, shape.part(44).y + y_plus_right], [shape.part(46).x + x_plus_bottom, shape.part(46).y - y_plus_right], 
            [shape.part(46).x + x_plus_bottom, shape.part(46).y - y_plus_right], [shape.part(41).x - x_plus_bottom, shape.part(41).y - y_plus_right],
            [shape.part(41).x - x_plus_bottom, shape.part(41).y - y_plus_left], [shape.part(37).x + x_plus_top, shape.part(37).y + y_plus_left]
            ], np.int32)
    
    pts = eye_contour.reshape((-1, 1, 2))
    #im_res = cv2.polylines(im_res, [pts], True, (199, 199, 199, 255), 1)
    cv2.fillPoly(im_res, [pts], color = (118, 113, 168, 255))

def draw_pupils(shape, im_res, color=(61, 71, 118, 255)):

    radius = int(abs(shape.part(39).x - shape.part(36).x)/4)
    pupil_x = int((abs(shape.part(39).x + shape.part(36).x)) / 2)
    pupil_y = int((abs(shape.part(39).y + shape.part(36).y)) / 2)
    pupil = (pupil_x, pupil_y)
    cv2.circle(im_res, pupil, radius, color, thickness=-1) #-1 means fill

    radius = int(abs(shape.part(42).x - shape.part(45).x)/4)
    pupil_x = int((abs(shape.part(42).x + shape.part(45).x)) / 2)
    pupil_y = int((abs(shape.part(42).y + shape.part(45).y)) / 2)
    pupil = (pupil_x, pupil_y)
    cv2.circle(im_res, pupil, radius, color, thickness=-1) #-1 means fill
 

################
# END FACE
################
def masks2anim(img_masks_opencv, img_real_opencv, addFace, shadowSize):
    #find relevant contours
    contours_not_dilated, contours_raw, colors = getContours(img_masks_opencv)

    #simplify
    contours_simplified = simplifyContours(contours_raw)

    #draw contours, pixelate and write file
    imcolor = np.zeros_like(img_masks_opencv)
    imcolor = addAlpha(imcolor)
    imcolor_result = fillContours(contours_simplified, colors, imcolor)
    
    #add shadows
    imcolor_result = addShadow(imcolor_result, shadowSize)

    #draw face elements
    if addFace:
        face(img_real_opencv, imcolor_result)
    
    return imcolor_result



