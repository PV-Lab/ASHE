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

# -- local imports --
# Adjust hyper parameters internally in Detect_Glass & Detect_Aluminum
import Detect_Glass as glass
import Detect_Alum as alum

# Prediction Parameters
NUM_SAMPLES = 100 # Number of valid frames to average over

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
        #print("overlap", overlap_percentage)

        return overlap_percentage

    return -1


def slant_diff(alum, glass):

    alum_slope = min(slope(alum[0], alum[1]), slope(alum[1], alum[2]))
    glass_slope = min(slope(glass[0], glass[1]), slope(glass[1], glass[2]))

    slope_diff = abs(alum_slope - glass_slope)

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

    return (overlap > overlap_threshold and slant > slant_threshold) or (len(alum_sampling) < 50 or len(glass_sampling) < 50)


def predict_single_static(aluminum, glass, crop_window, overlap_threhold, slant_threshold):
    alum_rect = alum_find_corners(img_alum, INTERNAL_CROP_RECT_TOP, save_prefix="doc_alum")
    glass_rect = glass_find_corners(img_glass, INTERNAL_CROP_RECT_TOP, save_prefix="doc_glass")
    overlap, slant = compare_rectangles(alum_rect, glass_rect), slant_diff(alum_rect, glass_rect)
    return overlap < overlap_threhold and slant < slant_threshold

if __name__ == "__main__":


    img_alum = cv2.imread(alum_img_path)
    img_glass = cv2.imread(glass_img_path)


    # Format: (x_start, y_start, width, height) - Relative to camera frame
    INTERNAL_CROP_RECT_TOP = (900, 260, 380, 250) # User confirmed crop bounds
    INTERNAL_CROP_RECT_BOT = (900, 540, 380, 250) # User confirmed crop bounds

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        print("Warning: No color frame received.")

    # Convert images to numpy arrays
    frame_bgr = np.asanyarray(color_frame.get_data())


    top = predict(pipeline, INTERNAL_CROP_RECT_TOP, 0.9, 0.5)
    bot = predict(pipeline, INTERNAL_CROP_RECT_BOT, 0.9, 0.5)


    # DEBUG CROP WINDOW WITH THIS DISPLAY:
    cv2.rectangle(frame_bgr, (900, 260), (930 + 380, 260 + 250), (0, 255, 255), 2)
    cv2.rectangle(frame_bgr, (900, 540), (930 + 380, 540 + 250), (0, 255, 255), 2)
    cv2.imshow("display", frame_bgr)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quit key pressed.")
        break
    #inp = input("Go?")
