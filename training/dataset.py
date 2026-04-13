# src/dataset_img.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # For image transformations
from PIL import Image # Python Imaging Library for loading images
import pandas as pd
import numpy as np

class ImageEdgeDataset(Dataset):
    """
    PyTorch Dataset for loading cropped edge images and their labels.
    Applies necessary transformations for CNN input.
    """
    def __init__(self, image_dir, label_file, image_transform=None, target_transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing cropped image files.
            label_file (str): Path to the CSV file mapping image filenames to labels
                              (must contain 'filename' and 'label' columns).
            image_transform (callable, optional): Transformations to apply to the image
                                                 (e.g., resize, to tensor, normalize).
            target_transform (callable, optional): Transformations to apply to the label.
        """
        self.image_dir = image_dir
        self.label_file = label_file
        self.image_transform = image_transform
        self.target_transform = target_transform

        # --- Load Label Mapping ---
        try:
            if not os.path.exists(self.label_file):
                raise FileNotFoundError(f"Label file not found: {self.label_file}")
            label_df = pd.read_csv(self.label_file)
            if 'filename' not in label_df.columns or 'label' not in label_df.columns:
                raise ValueError("Label file needs 'filename', 'label' columns.")
            # Ensure filename column is string type
            label_df['filename'] = label_df['filename'].astype(str)
            self.label_dict = dict(zip(label_df['filename'], label_df['label']))
            print(f"Loaded labels for {len(self.label_dict)} images from {self.label_file}")
        except Exception as e:
            raise IOError(f"Error reading label file {self.label_file}: {e}")

        # --- Collect Image Filenames present in the label file ---
        try:
            if not os.path.isdir(self.image_dir):
                raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
            all_files_in_folder = set(os.listdir(self.image_dir))
            # Filter filenames from label dict that actually exist in the folder
            # Assumes labels file uses base filenames without path
            self.image_filenames = sorted([
                fname for fname in self.label_dict.keys() if fname in all_files_in_folder
            ])
            if not self.image_filenames:
                raise ValueError(f"No image files found in {self.image_dir} match the filenames listed in {self.label_file}.")
            print(f"Found {len(self.image_filenames)} matching image files listed in label file.")

        except Exception as e:
            raise IOError(f"Error listing or matching files in {self.image_dir}: {e}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Loads image and label for the given index, applies transforms.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor
                   and label is the corresponding label.
        """
        # Get filename and construct full path
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        # Load image using PIL
        try:
            # Ensure image is loaded in RGB mode, even if grayscale
            # (CNN often expects 3 channels unless specified otherwise)
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            # Return dummy data or raise error
            # Creating dummy data matching expected output shape after transforms
            # This is tricky, depends on transforms. Better to ensure data exists.
            # For now, let's return None and handle in training loop or raise error
            # raise FileNotFoundError(f"Image file not found at {img_path}")
            return None, None # Or handle appropriately
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None, None # Or handle appropriately


        # Get label from dictionary
        label = self.label_dict.get(img_filename)
        if label is None:
            print(f"Warning: Label not found for {img_filename}. Assigning default 0.")
            label = 0 # Assign a default label or handle error

        # Apply image transformations (if any)
        if self.image_transform:
            try:
                image = self.image_transform(image)
            except Exception as e:
                 print(f"Error applying image transform to {img_filename}: {e}")
                 # Handle error, maybe return None
                 return None, None

        # Apply target/label transformations (if any)
        if self.target_transform:
            label = self.target_transform(label)

        # Ensure label is a tensor (CrossEntropyLoss expects LongTensor)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label).long()

        return image, label

# --- Example Usage ---

if __name__ == '__main__':

    # Define paths (replace with your actual paths)
    PROJECT_ROOT = "." # Assuming running from project root
    DATA_DIR = os.path.join(PROJECT_ROOT, "data_img") # Example data dir
    IMAGE_DIR = os.path.join(DATA_DIR, "processed_crops") # Contains cropped images
    LABEL_FILE = os.path.join(DATA_DIR, "image_labels.csv") # Maps image filenames to labels

    # --- Create Dummy Data for Demonstration ---
    print("Creating dummy data for demonstration...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    dummy_labels = []
    img_size_demo = 96
    for i in range(20): # Create 20 dummy images
        img_name = f"crop_{i:03d}.png"
        # Create a simple dummy image (e.g., random noise)
        dummy_img_array = np.random.randint(0, 256, size=(img_size_demo, img_size_demo, 3), dtype=np.uint8)
        dummy_img = Image.fromarray(dummy_img_array)
        dummy_img.save(os.path.join(IMAGE_DIR, img_name))
        # Assign dummy label (alternating 0 and 1)
        dummy_labels.append({'filename': img_name, 'label': i % 2})
    # Create dummy labels CSV
    pd.DataFrame(dummy_labels).to_csv(LABEL_FILE, index=False)
    print("Dummy data created.")
    # --- End Dummy Data Creation ---


    # --- Define Transformations ---
    # 1. Resize image to the size your CNN expects
    # 2. Convert PIL Image to PyTorch Tensor (scales pixels to [0, 1])
    # 3. Normalize pixel values (using ImageNet mean/std is common, adjust if needed)
    input_cnn_size = 96 # Should match the input_size of your EdgeClassifierCNN
    # Example using ImageNet stats (adjust if your data distribution is very different)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define separate transforms for training (with augmentation) and validation (no augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((input_cnn_size, input_cnn_size)),
        # --- Add Data Augmentation Here ---
        transforms.RandomHorizontalFlip(p=0.5), # Example: Randomly flip horizontally
        transforms.RandomRotation(degrees=5), # Example: Randomly rotate +/- 5 degrees
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Example: Adjust brightness/contrast
        # ---------------------------------
        transforms.ToTensor(), # Converts PIL image (H, W, C) [0, 255] to Tensor (C, H, W) [0, 1]
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std) # Standardize using mean/std
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((input_cnn_size, input_cnn_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Create Datasets ---
    print("\nCreating datasets...")
    # Assume we use the same data for train/val split here for simplicity
    # In practice, you'd split your data/labels file beforehand
    full_dataset_filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')] # Get all image names
    # Create a temporary full labels df for splitting (replace with your actual splitting logic)
    temp_full_label_df = pd.read_csv(LABEL_FILE)
    temp_full_label_df = temp_full_label_df[temp_full_label_df['filename'].isin(full_dataset_filenames)]

    # Simple random split (replace with stratified split if needed)
    train_df = temp_full_label_df.sample(frac=0.8, random_state=42)
    val_df = temp_full_label_df.drop(train_df.index)

    # Save temporary split label files
    train_label_file = os.path.join(DATA_DIR, "train_labels.csv")
    val_label_file = os.path.join(DATA_DIR, "val_labels.csv")
    train_df.to_csv(train_label_file, index=False)
    val_df.to_csv(val_label_file, index=False)

    # Create dataset instances using the split label files and appropriate transforms
    train_dataset = ImageEdgeDataset(
        image_dir=IMAGE_DIR,
        label_file=train_label_file,
        image_transform=train_transforms
    )
    val_dataset = ImageEdgeDataset(
        image_dir=IMAGE_DIR,
        label_file=val_label_file,
        image_transform=val_transforms
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # --- Create DataLoaders ---
    print("\nCreating DataLoaders...")
    batch_size = 4 # Example batch size
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data
        num_workers=2 # Use multiple processes to load data (adjust based on system)
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=2
    )
    print(f"Train DataLoader: {len(train_loader)} batches")
    print(f"Validation DataLoader: {len(val_loader)} batches")

    # --- Test DataLoader ---
    print("\nTesting DataLoader...")
    try:
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"  Batch image shape: {images.shape}") # Should be [batch_size, channels, height, width]
        print(f"  Batch label shape: {labels.shape}")   # Should be [batch_size]
        print(f"  Batch labels: {labels}")
        print("DataLoader test successful!")
    except Exception as e:
        print(f"Error testing DataLoader: {e}")

    # Clean up dummy data
    # print("\nCleaning up dummy data...")
    # shutil.rmtree(DATA_DIR)
    # print("Dummy data removed.")