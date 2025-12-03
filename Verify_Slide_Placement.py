import numpy as np
import pyrealsense2 as rs
import cv2
import time

# --- Local Imports ---
import CNN_Predict as model
import Deterministic_Predict as det

# -- hyper parameters --

# Format: (x_start, y_start, width, height) - Relative to camera frame
INTERNAL_CROP_RECT_TOP = (900, 260, 380, 250) # User confirmed crop bounds
INTERNAL_CROP_RECT_BOT = (900, 540, 380, 250) # User confirmed crop bounds


# --- Main Execution ---
if __name__ == "__main__":


    # CAMERA INITIALIZE
    pipeline = None
    try:

        while True:
            print("Configuring RealSense camera...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            print("Starting pipeline...")
            profile = pipeline.start(config)
            print("Pipeline started. Basline Front First then Back.")
            # Allow camera to stabilize
            time.sleep(2)

            deterministic_top = det.predict(pipeline, INTERNAL_CROP_RECT_TOP, 0.9, 0.5)
            deterministic_bot = det.predict(pipeline, INTERNAL_CROP_RECT_BOT, 0.9, 0.5)
            bot, top = model.return_predictions(pipeline)

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("Warning: No color frame received.")

            # Convert images to numpy arrays
            frame_bgr = np.asanyarray(color_frame.get_data())

            #print(f"det: {deterministic_top}, {deterministic_bot}")
            #print(f"mod: {top}, {bot}")

            top_text = "Success" if deterministic_top and "Success" in top else "Failure"
            bot_text = "Success" if deterministic_bot and "Success" in bot else "Failure"

            print("FRONT SLIDE PLACEMENT: ", top_text)
            print("BACK SLIDE  PLACEMENT: ", bot_text)

            cv2.putText(frame_bgr, top_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame_bgr, bot_text, (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # Show the frame
            cv2.imshow('Live Inference - Press Q to Quit', frame_bgr)
            
            next = input("NEXT (Y/N)?")

            if next in "nN":
                quit()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed.")
                exit()

    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        if pipeline:
            try: pipeline.stop()
            except: pass
        exit()


    
