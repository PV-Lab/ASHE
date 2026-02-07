# src/live_predict_cnn.py
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import argparse
import os
import time
from PIL import Image
from torchvision import transforms

# --- Local Imports ---
# Assuming model_cnn.py is in the same directory or path is set
try:
    from model_cnn import EdgeClassifierCNN # Import the CNN model class
except ImportError:
    print("Error: Could not import EdgeClassifierCNN.")
    print("Ensure model_cnn.py is in the same directory or accessible in PYTHONPATH.")
    exit()

# --- Configuration ---

# ROI Definition (MUST MATCH preprocessing script)
ROI_X = 870
ROI_Y = 260
ROI_WIDTH = 130
ROI_HEIGHT = 265
ROI_OFFSET = 280

# CNN Input Size (MUST MATCH model definition and preprocessing)
CNN_INPUT_SIZE = 96

# Model Configuration
NUM_CLASSES = 2
INPUT_CHANNELS = 3 # Assuming RGB input for the CNN

# Transformation for inference (Resize, ToTensor, Normalize)
# Use the same normalization stats as training (e.g., ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# --- Helper Functions ---

def crop(x, y, width, height, im):
    """ Crops image with boundary checks """
    h_im, w_im = im.shape[:2]
    y_start = max(y, 0); x_start = max(x, 0)
    y_end = min(y + height, h_im); x_end = min(x + width, w_im)
    if y_end <= y_start or x_end <= x_start: return None # Return None if crop is invalid
    return im[y_start:y_end, x_start:x_end]

def preprocess_live_frame(frame_bgr, side="BOT"):
    """
    Crops, converts to PIL, applies transforms for CNN input.

    Args:
        frame_bgr (np.ndarray): Input frame in BGR format from camera.

    Returns:
        torch.Tensor: Preprocessed tensor ready for the model (1, C, H, W),
                      or None if cropping fails.
    """
    # 1. Crop
    offset = ROI_OFFSET if side == "BOT" else 0

    img_crop = crop(ROI_X, ROI_Y + offset, ROI_WIDTH, ROI_HEIGHT, frame_bgr)
    if img_crop is None or img_crop.size == 0:
        # print("Warning: Crop resulted in empty image.")
        return None

    # 2. Convert BGR (OpenCV) to RGB (PIL)
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 3. Apply Transforms
    try:
        input_tensor = inference_transform(img_pil)
        # Add batch dimension -> (1, C, H, W)
        input_tensor = input_tensor.unsqueeze(0)
        return input_tensor
    except Exception as e:
        print(f"Error applying transforms: {e}")
        return None

def load_model(model_path, device):
    """
    Loads the trained CNN model.
    """
    #print("Loading model architecture...")
    model = EdgeClassifierCNN(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        input_size=CNN_INPUT_SIZE
    )
    #print(f"Loading model weights from: {model_path}")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
        # Load state dict, ensuring it's mapped to the correct device
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device) # Move model to device
        model.eval() # Set model to evaluation mode!
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None


def predict(input_tensor, model):

    label_map = {0: "Failure", 1: "Success"}
    text_color_map = {0: (0, 0, 255), 1: (0, 255, 0)} # Red for Failure, Green for Success

    if input_tensor is not None:

        # Perform Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        pred_label_idx = predicted_idx.item()
        pred_label_name = label_map.get(pred_label_idx, "Unknown")
        pred_confidence = confidence.item()

        prediction_text = f"{pred_label_name} ({pred_confidence:.2f})"
        text_color = text_color_map.get(pred_label_idx, (255, 255, 255)) # Default white
    else:
        prediction_text = "ERROR: Crop Failed"
        text_color = (0, 0, 255) # Red for error

    return prediction_text, text_color

def predict_single_static(frame, model, device):
    # Load input
    bot_input_tensor = preprocess_live_frame(frame, "BOT")
    bot_input_tensor =  bot_input_tensor.to(device)


    # prediction pipeline
    bot_pred_text, _ = predict(bot_input_tensor, model)

    return bot_pred_text


def return_predictions(pipeline, bot_mod=".\\cnn_checkpoints\\cnn_best_model_epoch_50_acc_100.00.pth", top_mod=".\\cnn_checkpoints\\cnn_best_top_model_epoch_50_acc_100.00.pth"):

    # Get Picture
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Warning: No color frame received.")
        return "FAILURE TO GET IMAGE", "STOP"

    # Convert images to numpy arrays
    frame_bgr = np.asanyarray(color_frame.get_data())


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    bot_model = load_model(bot_mod, device)
    top_model = load_model(top_mod, device)

    # Load input
    bot_input_tensor = preprocess_live_frame(frame_bgr, "BOT")
    top_input_tensor = preprocess_live_frame(frame_bgr, "TOP")

    bot_input_tensor =  bot_input_tensor.to(device)
    top_input_tensor =  top_input_tensor.to(device)


    # prediction pipeline
    bot_pred_text, _= predict(bot_input_tensor, bot_model)
    top_pred_text, _ = predict(top_input_tensor, top_model)

    return bot_pred_text, top_pred_text


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live inference using trained Edge Classifier CNN')
    parser.add_argument('--bot_model_path', type=str, required=True,
                        help='Path to the saved model checkpoint (.pth file)')
    parser.add_argument('--top_model_path', type=str, required=True,
                        help='Path to the saved model checkpoint (.pth file)')
    parser.add_argument('--width', type=int, default=1920, help='Camera frame width')
    parser.add_argument('--height', type=int, default=1080, help='Camera frame height')
    parser.add_argument('--fps', type=int, default=30, help='Camera frame rate')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA inference')

    args = parser.parse_args()

    # --- Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    bot_model = load_model(args.bot_model_path, device)
    top_model = load_model(args.top_model_path, device)
    if bot_model is None or top_model is None:
        print("Exiting due to model loading failure.")
        exit()

    # --- Configure and Start Camera ---
    pipeline = None
    try:
        print("Configuring RealSense camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        print("Starting pipeline...")
        profile = pipeline.start(config)
        print("Pipeline started.")
        # Allow camera to stabilize
        time.sleep(2)
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        if pipeline:
            try: pipeline.stop()
            except: pass
        exit()

    # --- Inference Loop ---
    print("\nStarting live inference loop... Press 'q' to quit.")
    label_map = {0: "Failure", 1: "Success"}
    text_color_map = {0: (0, 0, 255), 1: (0, 255, 0)} # Red for Failure, Green for Success

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Warning: No color frame received.")
                continue

            # Convert images to numpy arrays
            frame_bgr = np.asanyarray(color_frame.get_data())

            # Preprocess the frame for the CNN
            bot_input_tensor = preprocess_live_frame(frame_bgr, "BOT")
            top_input_tensor = preprocess_live_frame(frame_bgr, "TOP")

            bot_input_tensor =  bot_input_tensor.to(device)
            top_input_tensor =  top_input_tensor.to(device)

            bot_pred_text = "Processing..."
            text_color = (255, 255, 0) # Yellow for processing/error

            bot_pred_text, text_color = predict(bot_input_tensor, bot_model)
            top_pred_text, top_text_color = predict(top_input_tensor, top_model)



            # --- Display ---
            # Draw ROI rectangle on the main frame for visualization
            # Put prediction text on the main frame

            # bot pred
            cv2.rectangle(frame_bgr, (ROI_X, ROI_Y + ROI_OFFSET), (ROI_X + ROI_WIDTH, ROI_Y + ROI_OFFSET + ROI_HEIGHT), (0, 255, 255), 2) # Yellow ROI box
            cv2.putText(frame_bgr, bot_pred_text, (ROI_X + 5, ROI_Y + ROI_OFFSET + 25), # Position near top-left of ROI
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            # top pred
            cv2.rectangle(frame_bgr, (ROI_X, ROI_Y), (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), (0, 255, 255), 2) # Yellow ROI box
            cv2.putText(frame_bgr, top_pred_text, (ROI_X + 5, ROI_Y + 25), # Position near top-left of ROI
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, top_text_color, 2)

            # Show the frame
            cv2.imshow('Live Inference - Press Q to Quit', frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed.")
                break

    finally:
        # Stop streaming
        if pipeline:
            print("Stopping pipeline...")
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Windows closed. Exiting.")
