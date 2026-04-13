#!/usr/bin/env python3
"""
Implementation of determinstic methods of predicting successful
glass placement. Uses overlap and slant to make a prediction as
to if a glass is successfully placed
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from shapely import Polygon
import math

# -- local imports --
# Adjust hyper parameters internally in Detect_Glass & Detect_Aluminum
import detect_glass as glass
import detect_alum as alum

# Prediction Parameters
NUM_SAMPLES = 100 # Number of valid frames to average over

def repeated_sample_static(obj, image, crop_window, num_samples=NUM_SAMPLES):

    samples = []
    # Apply gamma correction
    brightened_image_gamma = adjust_gamma(image, gamma=2.5)

    corners = obj.find_corners(brightened_image_gamma, crop_window)

    if corners is not None:
        samples.append(corners)
    
    return samples

# Function to apply gamma correction
def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def repeated_sample(obj, pipeline, crop_window, num_samples=NUM_SAMPLES):

    samples = []
    for _ in range(num_samples):

        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if not frame:
            print("Warning: No color frame received.")
            return "FAILURE TO GET IMAGE"

        image = np.asanyarray(frame.get_data())
        corners = obj.find_corners(image, crop_window)

        if corners is not None:
            samples.append(corners)

    return samples

# COMPARING DATA

def slope(left, right):
    m = (left[1] - right[1])/ max((left[0] - right[0], 1))
    return m

def compare_rectangles(alum, glass):

    """
    takes two arrays of bounding points

    returns percent of overlap
    """

    # check area overlap

    if alum is not None and glass is not None:
        poly1 = Polygon(alum)
        poly2 = Polygon(glass)

        if not poly1.is_valid or not poly2.is_valid:
            return -1

        intersection = poly1.intersection(poly2)

        if intersection.is_empty:
            return -1

        overlap_percentage = (intersection.area / poly1.area) * 100
        # print("overlap", overlap_percentage)

        return overlap_percentage

    return -1


def get_angle(p1, p2):
    # Returns the angle in degrees
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def slant_diff(alum, glass):
    # Get the angle of the first side for both
    # (Using % 90 helps normalize rectangles so it doesn't matter 
    # if you're looking at the long or short side)
    alum_angle = get_angle(alum[0], alum[1]) % 90
    glass_angle = get_angle(glass[0], glass[1]) % 90
    slope_diff = abs(alum_angle - glass_angle)
    # print(f"slope diff: {slope_diff}")

    return slope_diff


def averaged_sampling(alum_samples, glass_samples):

    """
    Take two arrays of samples;
    In each, a bounding rectangle:
    - list of tuples of corner points

    Returns the average overlap over (ideally 1000 samples)
    """

    area_sum = 0
    slope_sum = 0
    total = 0

    # compare each sample in the collected samples
    for poly1, poly2 in zip(alum_samples, glass_samples):

        to_add = compare_rectangles(poly1, poly2)
        to_add_slope = slant_diff(poly1, poly2)

        if to_add_slope > 45:
            to_add_slope = abs(90 - to_add_slope)

        if to_add != -1:
            area_sum += to_add
            slope_sum += to_add_slope
            total += 1

    if total != 0:
        return area_sum/total, slope_sum/total

    return 0, 0

def predict(pipeline, crop_window, overlap_threshold, slant_threshold):

    go = input("Press enter to sample aluminum.")

    alum_sampling = repeated_sample(alum, pipeline, crop_window)

    # ADD PAUSE
    go = input('Alum baseline done. Press enter to sample glass.')

    glass_sampling = repeated_sample(glass, pipeline, crop_window)
    overlap, slant = averaged_sampling(alum_sampling, glass_sampling)

    #print(f"overlap: {overlap}, slant: {slant}, num samples: {len(alum_sampling)} {len(glass_sampling)}")

    return (overlap > overlap_threshold and slant < slant_threshold) #or (len(alum_sampling) < 50 or len(glass_sampling) < 50)


def predict_single_static(img_alum, img_glass, crop_window, overlap_threshold, slant_threshold):
    alum_rect = repeated_sample_static(alum, img_alum, crop_window)
    # print(alum_rect)
    glass_rect = repeated_sample_static(glass, img_glass, crop_window)
    # print(glass_rect)
    # overlap, slant = compare_rectangles(alum_rect, glass_rect), slant_diff(alum_rect, glass_rect)
    overlap, slant = averaged_sampling(alum_rect, glass_rect)
    # print(f"overlap: {overlap}")
    # print(f"slant: {slant}")
    return alum_rect, glass_rect, overlap > overlap_threshold and slant < slant_threshold

if __name__ == "__main__":
    inp = "y"
    print("Configuring RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    print("Starting pipeline...")
    profile = pipeline.start(config)
    print("Pipeline started.")
    # Allow camera to stabilize
    time.sleep(2)

    while True:

        # Format: (x_start, y_start, width, height) - Relative to camera frame
        INTERNAL_CROP_RECT_TOP = (900, 260, 380, 250) # User confirmed crop bounds
        INTERNAL_CROP_RECT_BOT = (900, 540, 380, 250) # User confirmed crop bounds

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("Warning: No color frame received.")

        # Convert images to numpy arrays
        frame_bgr = np.asanyarray(color_frame.get_data())


        #top = predict(pipeline, INTERNAL_CROP_RECT_TOP, 0.9, 0.5)
        #bot = predict(pipeline, INTERNAL_CROP_RECT_BOT, 0.9, 0.5)


        # DEBUG CROP WINDOW WITH THIS DISPLAY:
        cv2.rectangle(frame_bgr, (900, 260), (930 + 380, 260 + 250), (0, 255, 255), 2)
        cv2.rectangle(frame_bgr, (900, 540), (930 + 380, 540 + 250), (0, 255, 255), 2)
        cv2.imshow("display", frame_bgr)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed.")
            break
        #inp = input("Go?")
