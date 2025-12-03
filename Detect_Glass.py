import cv2
import numpy as np
from shapely.geometry import Polygon
import pyrealsense2 as rs
import os # Keep os for potential future use, though not strictly needed now
import Image_Proccessing as im


# --- HYPER PARAMETERS ---
GLASS_WIDTH = 330
GLASS_HEIGHT = 215
GLASS_AREA = GLASS_WIDTH * GLASS_HEIGHT
GLASS = {"width": GLASS_WIDTH, "height": GLASS_HEIGHT, "area": GLASS_AREA}

# Define the hardcoded crop rectangle used internally by processing functions

def get_glass_edges(color_filtered_im):
    """ 
    Detect & clean edges using original Canny + Morph 
    """

    # bad image
    if color_filtered_im is None or color_filtered_im.size == 0:
        return np.zeros((1,1), dtype=np.uint8)

    # Canny morphology to isolate relevant edges
    edges = cv2.Canny(color_filtered_im, 50, 300, apertureSize=3)

    kernel = np.ones((3, 3), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cleaned_edges = cv2.morphologyEx(cleaned_edges, cv2.MORPH_OPEN, kernel)

    kernel_erode = np.ones((1, 1), np.uint8)
    cleaned_edges = cv2.erode(cleaned_edges, kernel_erode)

    return cleaned_edges

def edge_points(edges):
    """ 
    Use HoughLinesP to find line segments and return their endpoints
    """
    points = []

    # bad input
    if edges is None or edges.ndim != 2 or edges.dtype != np.uint8:
         return None

    # Original Hough parameters
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)

    if lines is not None:
        kept_lines_count = 0

        # keep only relevant lines
        for line_arr in lines:
            if im.is_considerable_line(line_arr, min_length=30): 
                x1, y1, x2, y2 = line_arr[0]
                points.append((x1, y1))
                points.append((x2, y2))
                kept_lines_count += 1

        if points:
            return np.array(points, dtype=np.int32)

    # no considerable lines found
    return None

def get_glass_data(color_image, crop_window):
    """
    Process image to get glass edges within the internally defined crop
    """
    x, y, w, h = crop_window # Use hardcoded crop
    cropped_im = im.crop(x, y, w, h, color_image)

    # bad input
    if cropped_im.size == 0:
        return np.zeros((1,1), dtype=np.uint8), (x,y), None

    blue_im = im.extract_blue(cropped_im)
    edges = get_glass_edges(blue_im)

    return edges

def find_corners(color_image, crop_window):
    """ 
    Find corners of the glass object using internal crop and processing
    (Hough lines -> bounding rect)
    """
    
    glass_edges = get_glass_data(color_image, crop_window)
    glass = edge_points(glass_edges)

    if glass is not None and len(glass) > 2:      
        all_points = np.vstack(glass)  # Stack all contour points together
        epsilon = 0.001 * cv2.arcLength(all_points, True)  # Approximation factor
        polygon = cv2.approxPolyDP(all_points, epsilon, True)  # Approximate polygon

        # Apply Convex Hull to make sure it's convex
        hull = cv2.convexHull(polygon)
        points = im.bounding_rect(hull)

        if im.is_considerable_shape(points, GLASS):
            return points

    return None
