# preprocess_cnn_data.py
import cv2
import numpy as np
import os
import glob
import random
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---

# ROI Definition (Base ROI - will be shifted for augmentation)
# ACTION REQUIRED: Set your actual ROI coordinates
ROI_X = 870
ROI_Y = 260
ROI_WIDTH = 130
ROI_HEIGHT = 265
ROI_OFFSET = 280

# Augmentation Settings
NUM_AUGMENTATIONS_PER_IMAGE = 15 # How many augmented versions to create per original image

# Augmentation Parameters (Adjust these ranges)
# --- Reduced Lighting Variations ---
AUG_BRIGHTNESS_RANGE = (-10, 10) # Smaller brightness adjustment
AUG_CONTRAST_RANGE = (0.95, 1.05) # Smaller contrast adjustment
# --- Other Augmentations ---
AUG_ROTATION_DEG = (-1, 1)      # Degrees (Keep small)
AUG_SCALE_RANGE = (0.98, 1.02)  # Zoom in/out slightly (Keep small)
AUG_GAUSSIAN_NOISE_STD = (0, 5) # Reduced max noise std dev
# --- Translation via Crop Shift ---
AUG_CROP_SHIFT_PX = 5           # Max pixels to shift crop window up/down/left/right

# Train/Validation Split Ratio
VAL_SPLIT_RATIO = 0.2 # Use 20% of the data for validation

# --- Helper Functions ---

def crop(x, y, width, height, im):
    """ Crops image with boundary checks """
    h_im, w_im = im.shape[:2]
    y_start = max(y, 0); x_start = max(x, 0)
    y_end = min(y + height, h_im); x_end = min(x + width, w_im)
    if y_end <= y_start or x_end <= x_start:
        # print(f"Warning: Invalid crop dimensions calculated ({x_start},{y_start}) -> ({x_end},{y_end}).")
        return np.zeros((0, 0, im.shape[2]) if len(im.shape) == 3 else (0, 0), dtype=im.dtype)
    return im[y_start:y_end, x_start:x_end]

# --- Augmentation Functions ---

def augment_image_after_crop(img):
    """
    Applies a set of random augmentations (excluding translation via crop)
    to an already cropped image.
    """
    rows, cols = img.shape[:2]
    if rows == 0 or cols == 0: return img # Skip if crop was empty
    augmented_img = img.copy()

    # 1. Brightness / Contrast
    brightness = random.randint(*AUG_BRIGHTNESS_RANGE)
    contrast = random.uniform(*AUG_CONTRAST_RANGE)
    augmented_img = cv2.convertScaleAbs(augmented_img, alpha=contrast, beta=brightness)
    augmented_img = np.clip(augmented_img, 0, 255)

    # 2. Rotation
    angle = random.uniform(*AUG_ROTATION_DEG)
    center = (cols // 2, rows // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented_img = cv2.warpAffine(augmented_img, rot_mat, (cols, rows),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # 3. Scaling (Zoom)
    scale = random.uniform(*AUG_SCALE_RANGE)
    scale_mat = cv2.getRotationMatrix2D(center, 0, scale)
    augmented_img = cv2.warpAffine(augmented_img, scale_mat, (cols, rows),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # 4. Gaussian Noise
    noise_std = random.uniform(*AUG_GAUSSIAN_NOISE_STD)
    if noise_std > 0:
        # Ensure noise has same dtype as image before adding
        gauss = np.random.normal(0, noise_std, augmented_img.shape)
        # Add noise and clip carefully to avoid dtype issues
        augmented_img = np.clip(augmented_img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    return augmented_img

# --- Main Processing Function ---

def preprocess_and_augment(input_base_dir, output_base_dir):
    """
    Processes images from success/failure folders, applies shifted crops for translation,
    augments further, and saves to train/val directories. Generates label CSVs.
    """
    input_success_dir = os.path.join(input_base_dir, "success")
    input_failure_dir = os.path.join(input_base_dir, "failure")
    output_train_dir = os.path.join(output_base_dir, "train")
    output_val_dir = os.path.join(output_base_dir, "val")

    # --- Input Validation ---
    if not os.path.isdir(input_success_dir): print(f"Error: Input 'success' directory not found: {input_success_dir}"); return
    if not os.path.isdir(input_failure_dir): print(f"Error: Input 'failure' directory not found: {input_failure_dir}"); return

    # --- Create Output Directories ---
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    print(f"Output will be saved to: {output_train_dir} and {output_val_dir}")

    # --- Gather and Label Input Files ---
    success_files = [(os.path.join(input_success_dir, f), 1) for f in os.listdir(input_success_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    failure_files = [(os.path.join(input_failure_dir, f), 0) for f in os.listdir(input_failure_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    all_files = success_files + failure_files
    if not all_files: print("Error: No image files found in input success/failure directories."); return
    print(f"Found {len(success_files)} success images and {len(failure_files)} failure images.")
    random.shuffle(all_files)

    # --- Split into Train and Validation Sets ---
    train_files, val_files = train_test_split(
        all_files, test_size=VAL_SPLIT_RATIO, random_state=42, stratify=[label for _, label in all_files]
    )
    print(f"Splitting into {len(train_files)} training samples and {len(val_files)} validation samples.")

    # --- Process Files and Generate Labels ---
    train_labels_list = []
    val_labels_list = []

    # Helper function to process one file list (train or val)
    def process_file_list(file_list, output_dir, label_list, set_name):
        print(f"\nProcessing {set_name} set ({len(file_list)} images)...")
        skipped_load = 0
        skipped_crop = 0
        for img_path, label in tqdm(file_list, desc=f"Processing {set_name}"):
            try:
                img = cv2.imread(img_path)
                if img is None: skipped_load += 1; continue

                # Generate and save augmentations with shifted crops
                for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                    # Calculate random shift for this augmentation
                    dx = random.randint(-AUG_CROP_SHIFT_PX, AUG_CROP_SHIFT_PX)
                    dy = random.randint(-AUG_CROP_SHIFT_PX, AUG_CROP_SHIFT_PX)

                    # Apply shift to base ROI coordinates
                    current_roi_x = ROI_X + dx
                    current_roi_y = ROI_Y + dy

                    # Crop the *original* image with the *shifted* ROI
                    img_top = crop(current_roi_x, current_roi_y, ROI_WIDTH, ROI_HEIGHT, img)
                    img_bot = crop(current_roi_x, current_roi_y + ROI_OFFSET, ROI_WIDTH, ROI_HEIGHT, img)

                    if img_top.size == 0 or img_bot.size == 0:
                        # print(f"Warning: Shifted crop resulted in empty image for {img_path}, aug {i}. Skipping this aug.")
                        # Don't increment skipped_crop here, just skip this specific augmentation
                        continue

                    # Apply other augmentations (rotation, scale, noise, etc.) to the shifted crop
                    augmented_crop_top = augment_image_after_crop(img_top)
                    augmented_crop_bot = augment_image_after_crop(img_bot)

                    # Save the final augmented image
                    aug_filename_top = f"{os.path.splitext(os.path.basename(img_path))[0]}_top_aug_{i}.png" # Save as PNG
                    aug_filename_bot = f"{os.path.splitext(os.path.basename(img_path))[0]}_bot_aug_{i}.png" # Save as PNG
                    save_path_top = os.path.join(output_dir, aug_filename_top)
                    save_path_bot = os.path.join(output_dir, aug_filename_bot)
                    cv2.imwrite(save_path_top, augmented_crop_top)
                    cv2.imwrite(save_path_bot, augmented_crop_bot)
                    label_list.append({"filename": aug_filename_top, "label": label})
                    label_list.append({"filename": aug_filename_bot, "label": label})

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        print(f"{set_name} set: Skipped loading {skipped_load} original images.")


    # Process training files
    process_file_list(train_files, output_train_dir, train_labels_list, "Training")
    # Process validation files
    process_file_list(val_files, output_val_dir, val_labels_list, "Validation")

    # --- Save Label CSVs ---
    train_labels_df = pd.DataFrame(train_labels_list)
    val_labels_df = pd.DataFrame(val_labels_list)
    train_csv_path = os.path.join(output_base_dir, "train_labels.csv")
    val_csv_path = os.path.join(output_base_dir, "val_labels.csv")
    train_labels_df.to_csv(train_csv_path, index=False)
    val_labels_df.to_csv(val_csv_path, index=False)
    print(f"\nSaved training labels to: {train_csv_path} ({len(train_labels_df)} entries)")
    print(f"Saved validation labels to: {val_csv_path} ({len(val_labels_df)} entries)")
    print("\nPreprocessing and augmentation complete.")


# --- Command Line Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and augment images for CNN training.")
    parser.add_argument("input_dir", help="Base directory containing 'success' and 'failure' subdirectories with original images.")
    parser.add_argument("output_dir", help="Base directory where 'train' and 'val' subdirectories and label CSVs will be created.")
    args = parser.parse_args()
    input_base = os.path.abspath(args.input_dir)
    output_base = os.path.abspath(args.output_dir)

    print("\n--- Configuration Summary ---")
    print(f"Input Base Directory:  '{input_base}'")
    print(f"Output Base Directory: '{output_base}'")
    print(f"Base ROI (X,Y,W,H):    ({ROI_X}, {ROI_Y}, {ROI_WIDTH}, {ROI_HEIGHT})")
    print(f"Crop Shift Max Px:     {AUG_CROP_SHIFT_PX}")
    print(f"Augmentations/Image:   {NUM_AUGMENTATIONS_PER_IMAGE}")
    print(f"Validation Split:      {VAL_SPLIT_RATIO:.1%}")
    print("--- Augmentation Params ---")
    print(f"  Brightness: {AUG_BRIGHTNESS_RANGE}")
    print(f"  Contrast:   {AUG_CONTRAST_RANGE}")
    print(f"  Rotation:   {AUG_ROTATION_DEG} degrees")
    print(f"  Scale:      {AUG_SCALE_RANGE}")
    print(f"  Gauss Noise:{AUG_GAUSSIAN_NOISE_STD} std dev")
    print("-" * 27)

    confirm = input("Proceed with processing? (yes/no): ").lower()
    if confirm == 'yes':
        preprocess_and_augment(input_base, output_base)
    else:
        print("Operation cancelled.")

