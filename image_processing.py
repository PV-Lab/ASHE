import cv2
import numpy as np
import time
from shapely.geometry import Polygon
import pyrealsense2 as rs
import os # Keep os for potential future use, though not strictly needed now

# Image processing functions

def crop(x, y, width, height, im):
    """ 
    Crops image 
    """
    h_im, w_im = im.shape[:2]
    y_start = max(y, 0); x_start = max(x, 0)
    y_end = min(y + height, h_im); x_end = min(x + width, w_im)
    if y_end <= y_start or x_end <= x_start: return np.zeros((0, 0, im.shape[2]) if len(im.shape) == 3 else (0, 0), dtype=im.dtype)

    return im[y_start:y_end, x_start:x_end]

def extract_blue(color_im, lower_bound=[80, 0, 150], upper_bound=[120, 255, 250]):
    """ 
    Color mask an image (using passed in HSV values) 
    --> extract blue pixels 
    """

    lower_blue = np.asarray(lower_bound) 
    upper_blue = np.asarray(upper_bound)
    hsv_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3),np.float32)/9
    blurred_im = cv2.filter2D(hsv_im, -1, kernel)
    blue_mask = cv2.inRange(blurred_im, lower_blue, upper_blue)
    result = cv2.bitwise_and(color_im, color_im, mask= blue_mask)

    return result


def get_contours(image):
    """ 
    Get contours from image
    """

    # check image form
    if image is None or image.ndim != 2 or image.dtype != np.uint8:
       return []

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def find_largest_contour(contours):
    """ 
    Find largest contour based on arc length
    """

    largest_contour = None
    largest_contour_len = 0

    if contours:
        if len(contours) > 0:
            try:
                largest_contour_len = cv2.arcLength(contours[0], False)
                largest_contour = contours[0]
    
            except cv2.error as e:
                pass # Suppress warning (because we continue sampling)

        for contour in contours:
            try:
                size = cv2.arcLength(contour, False)
                if size > largest_contour_len:
                    largest_contour_len = size
                    largest_contour = contour

            except cv2.error as e:
                continue # Skip this contour
    
    return largest_contour


def is_considerable(contour, len_thresh=100):
    """
    Check if an identified contour is legitimate
      
    """

    if contour is None:
        return False

    try:
        perimeter = cv2.arcLength(contour, closed=False)
        return perimeter > len_thresh

    # bad contour
    except cv2.error as e:
        return False


def is_considerable_line(line, min_length=50):
    """ 
    Check if identified Hough line is legitimate
    """

    # check line form
    if line is None: 
        return False
    if not isinstance(line, (list, np.ndarray)) or len(line) == 0: 
        return False
    
    line_coords = line[0]
    if not isinstance(line_coords, (list, np.ndarray)) or len(line_coords) != 4: return False
    p1 = np.array(line_coords[:2]); p2 = np.array(line_coords[2:])
    length = np.linalg.norm(p1 - p2)
    return length > min_length



def clean_contours(contours): 
    """
    Filters contours
    """
    cleaned = [cnt for cnt in contours if is_considerable(cnt)]
    return cleaned


def bounding_rect(points):
    """
    Calculates minimum area bounding rectangle
    """
    if points is None or len(points) < 3:
        return None
    
    if points.ndim == 3 and points.shape[1] == 1: 
        np_points = points.reshape(-1, 2).astype(np.int32)

    elif points.ndim == 2 and points.shape[1] == 2: 
        np_points = points.astype(np.int32)

    else:
        return None

    try:
        rot_rect = cv2.minAreaRect(np_points)
        box_points = cv2.boxPoints(rot_rect)
        return box_points

    except cv2.error as e:
        return None
    
def is_considerable_shape(bounding_rectangle_corners, type_obj):
    """
    Determines if a given rectangle is considerable as 
    the type of object passed in

    Bounding_rect_corners: corners of rectangular shape
    Type: has desired dimensions of objects.
    
    """

    if bounding_rectangle_corners is not None and len(bounding_rectangle_corners) == 4:

        x1, y1 = bounding_rectangle_corners[0]
        x2, y2 = bounding_rectangle_corners[1]
        x3, y3 = bounding_rectangle_corners[2]

    
        side1_len = ((x1-x2)**2 + (y1-y2)**2) ** (1/2)
        side2_len = ((x2-x3)**2 + (y2-y3)**2) ** (1/2)

        height = side1_len
        width = side2_len

        is_valid = (
            (close_to(type_obj["height"], height) and close_to(type_obj["width"], width)) or
            (close_to(type_obj["height"], width) and close_to(type_obj["width"], height))
        )

        return is_valid

    # Bad input
    return False


def close_to(known, found):
    return known*0.9 < found < known*1.1

