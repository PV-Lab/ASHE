# src/train_cnn.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# Local imports (assuming train_cnn.py is in src/)
try:
    from dataset import ImageEdgeDataset # Import the image dataset class
    from model_cnn import EdgeClassifierCNN # Import the CNN model class
except ImportError:
    print("Error: Ensure dataset_img.py and model_cnn.py are in the same directory (src/) or run using 'python -m src.train_cnn'")
    exit()

def calculate_class_weights(label_file_path):
    """Calculates class weights for CrossEntropyLoss based on label file."""
    try:
        df = pd.read_csv(label_file_path)
        label_counts = df['label'].value_counts().sort_index()
        if len(label_counts) != 2:
            print(f"Warning: Expected 2 classes in {label_file_path}, found {len(label_counts)}. Not using weights.")
            return None
        # Formula: Total Samples / (Num Classes * Num Samples in Class)
        total_samples = len(df)
        num_classes = len(label_counts)
        weights = total_samples / (num_classes * label_counts.values)
        print(f"Calculated Class Weights: {weights.tolist()} for classes {label_counts.index.tolist()}")
        return torch.tensor(weights, dtype=torch.float)
    except FileNotFoundError:
        print(f"Warning: Label file {label_file_path} not found. Cannot calculate class weights.")
        return None
    except Exception as e:
        print(f"Warning: Error calculating class weights from {label_file_path}: {e}. Not using weights.")
        return None


def main(args):
    """Main training function."""

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoints directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Transformations ---
    # Define transformations needed for model input
    # Note: Heavy augmentation was done offline. These are mainly for resizing, tensor conversion, and normalization.
    # You *can* add light online augmentation (like flips) to train_transforms if desired.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        # transforms.RandomHorizontalFlip(p=0.5), # Optional: Add light online augmentation
        transforms.ToTensor(), # Converts PIL image [0, 255] to Tensor [0, 1]
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std) # Standardize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Data Loading ---
    print("Setting up DataLoaders...")
    try:
        train_image_dir = os.path.join(args.data_dir, "train")
        train_label_file = os.path.join(args.data_dir, "train_labels.csv")
        val_image_dir = os.path.join(args.data_dir, "val")
        val_label_file = os.path.join(args.data_dir, "val_labels.csv")

        if not os.path.isdir(train_image_dir): raise FileNotFoundError(f"Training image directory not found: {train_image_dir}")
        if not os.path.isfile(train_label_file): raise FileNotFoundError(f"Training label file not found: {train_label_file}")
        if not os.path.isdir(val_image_dir): raise FileNotFoundError(f"Validation image directory not found: {val_image_dir}")
        if not os.path.isfile(val_label_file): raise FileNotFoundError(f"Validation label file not found: {val_label_file}")

        train_dataset = ImageEdgeDataset(
            image_dir=train_image_dir,
            label_file=train_label_file,
            image_transform=train_transforms
        )
        val_dataset = ImageEdgeDataset(
            image_dir=val_image_dir,
            label_file=val_label_file,
            image_transform=val_transforms
        )

        # Adjust num_workers based on your system
        num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 0
        print(f"Using {num_workers} workers for DataLoader.")

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True # Drop last incomplete batch if dataset size not divisible by batch size
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
    except Exception as e:
        print(f"Error creating datasets or dataloaders: {e}")
        print(f"Please check data directory ('{args.data_dir}'), ensure 'train/' & 'val/' subdirs and label CSVs exist.")
        return # Exit if data loading fails

    # --- Model Initialization ---
    print("Initializing the CNN model...")
    try:
        # Assuming RGB input (3 channels)
        model = EdgeClassifierCNN(
            input_channels=3,
            num_classes=args.num_classes,
            input_size=args.input_size
        ).to(device)
        print("CNN model loaded onto device.")
        # print(model) # Uncomment to view model structure
    except Exception as e:
        print(f"Error initializing the model: {e}")
        return

    # --- Loss Function (with optional class weights) ---
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_label_file)
        if class_weights is not None:
            class_weights = class_weights.to(device) # Move weights to device

    criterion = nn.CrossEntropyLoss(weight=class_weights) # Pass weights (or None)
    print(f"Using CrossEntropyLoss {'with class weights' if class_weights is not None else 'without class weights'}.")


    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Consider ReduceLROnPlateau if validation loss plateaus
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.5) # Step decay

    # --- Training Loop ---
    best_val_acc = 0.0
    print("\nStarting training...")
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)

        for i, data in enumerate(progress_bar_train):
            images, labels = data
            if images is None or labels is None:
                print(f"Warning: Skipping batch {i} due to loading error.")
                continue
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) # CNN takes (B, C, H, W)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}, train batch {i}. Skipping.")
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            acc_train = (100.0 * correct_train / total_train) if total_train > 0 else 0.0
            progress_bar_train.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc_train:.2f}%'})

        epoch_train_loss = running_loss / total_train if total_train > 0 else 0.0
        epoch_train_acc = 100.0 * correct_train / total_train if total_train > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for data in progress_bar_val:
                images, labels = data
                if images is None or labels is None: continue
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected during validation epoch {epoch+1}. Skipping batch.")
                    continue

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_acc_so_far = (100.0 * correct_val / total_val) if total_val > 0 else 0.0
                progress_bar_val.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{val_acc_so_far:.2f}%'})

        epoch_val_loss = running_val_loss / total_val if total_val > 0 else 0.0
        epoch_val_acc = 100.0 * correct_val / total_val if total_val > 0 else 0.0

        # --- Logging and Checkpointing ---
        print(f"Epoch {epoch+1}/{args.epochs} Summary: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # Update learning rate scheduler
        scheduler.step()
        # If using ReduceLROnPlateau: scheduler.step(epoch_val_loss)

        # Save the model if validation accuracy improves
        if epoch_val_acc > (best_val_acc - 2) and total_val > 0:
            best_val_acc = epoch_val_acc
            save_path = os.path.join(args.checkpoint_dir, f'cnn_best_top_model_epoch_{epoch+1}_acc_{epoch_val_acc:.2f}.pth')
            try:
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best model saved to {save_path} (Val Acc: {best_val_acc:.2f}%)")
            except Exception as e:
                print(f"Error saving model checkpoint: {e}")

    print("="*30)
    print("Training finished!")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.2f}%")
    print(f"Model checkpoints are saved in the '{args.checkpoint_dir}' folder.")
    print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Edge Classifier CNN')

    # Data and Model paths
    parser.add_argument('--data_dir', type=str, default='./processed_data_top',
                        help="Path to the base directory containing train/val subdirs and label CSVs")
    parser.add_argument('--checkpoint_dir', type=str, default='./cnn_checkpoints',
                        help='Directory to save model checkpoints')

    # Model Hyperparameters
    parser.add_argument('--input_size', type=int, default=96,
                        help='Square size of input images for the CNN (e.g., 96, 128)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (e.g., 2 for success/failure)')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_step_size', type=int, default=15,
                        help='Step size for learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Optimizer weight decay')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Calculate and use class weights for loss function')

    # Execution settings
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')

    args = parser.parse_args()
    print("\n--- Training Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 26 + "\n")

    main(args)
