import cv2
import numpy as np
import time
from shapely.geometry import Polygon
import pyrealsense2 as rs
import os # Keep os for potential future use, though not strictly needed now
import Image_Proccessing as im

# --- HYPER PARAMETERS ---
ALUMINUM_WIDTH = 325
ALUMINUM_HEIGHT = 215
ALUMINUM_AREA = ALUMINUM_WIDTH * ALUMINUM_HEIGHT
ALUMINUM = {"width": ALUMINUM_WIDTH, "height": ALUMINUM_HEIGHT, "area": ALUMINUM_AREA}

def get_alum_edges(gray_im):
    """ 
    Detect edges & clean using original Canny + Morph 
    """

    # bad input
    if gray_im is None or gray_im.size == 0:
        return np.zeros((1,1), dtype=np.uint8)


    edges = cv2.Canny(gray_im, 30, 300, apertureSize=3)

    kernel_close = np.ones((3, 3), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)


    kernel_erode = np.ones((1, 1), np.uint8)
    cleaned_edges = cv2.erode(cleaned_edges, kernel_erode)

    return cleaned_edges

def get_alum_data(color_image, crop_window):
    """
    Process image to get aluminum contours within the internally defined crop
    """
    x, y, w, h = crop_window
    cropped_im = im.crop(x, y, w, h, color_image)

    # bad input
    if cropped_im.size == 0:
        return [], (x,y), None # Added None for consistent return type

    blurred_color = cv2.GaussianBlur(cropped_im, (3, 3), 0)
    gray_blurred = cv2.cvtColor(blurred_color, cv2.COLOR_BGR2GRAY)
    edges = get_alum_edges(gray_blurred)

    contours = im.get_contours(edges)
    cleaned_contours = im.clean_contours(contours) 

    return cleaned_contours


def find_corners(color_im, crop_window):
    """
    Find corners of the aluminum object using original internal crop and processing 
    """

    alum_contour = get_alum_data(color_im, crop_window)
    alum = im.find_largest_contour(alum_contour)

    if alum is not None:

        epsilon = 0.01 * cv2.arcLength(alum, False)  # Adjust for desired smoothness
        approx = cv2.approxPolyDP(alum, epsilon, True)

        # Apply Convex Hull to make sure it's convex
        hull = cv2.convexHull(approx)

        # approximate shape, ignore any small errors in clean edges
        hull = cv2.convexHull(approx)

        # bound hull to clean rectangle
        x, y, w, h = cv2.boundingRect(hull)

        bounding_rectangle = [(x, y), (x, y+h), (x+w, y+h),(x+w, y)]

        if im.is_considerable_shape(bounding_rectangle, ALUMINUM):
            return bounding_rectangle

    return None


