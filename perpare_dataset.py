import os
import glob
import shutil
import random

# --- CONFIGURATION ---
# Set the path to the folders containing all your original images and labels
SOURCE_IMAGES_DIR = "all_images"
SOURCE_LABELS_DIR = "all_labels"

# Set the path to the destination dataset folder
DEST_DATASET_DIR = "dataset"

# Set the percentage of data to be used for training (e.g., 0.8 for 80%)
TRAIN_SPLIT = 0.8


def create_dir_if_not_exists(path):
    """Helper function to create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_dataset():
    """
    Reads image and label files from source directories, splits them into
    train and validation sets, and copies them into the destination
    dataset folder structure.
    """
    print("Starting dataset preparation...")

    # Destination paths
    train_img_path = os.path.join(DEST_DATASET_DIR, "train", "images")
    train_lbl_path = os.path.join(DEST_DATASET_DIR, "train", "labels")
    val_img_path = os.path.join(DEST_DATASET_DIR, "val", "images")
    val_lbl_path = os.path.join(DEST_DATASET_DIR, "val", "labels")

    # Create destination directories
    create_dir_if_not_exists(train_img_path)
    create_dir_if_not_exists(train_lbl_path)
    create_dir_if_not_exists(val_img_path)
    create_dir_if_not_exists(val_lbl_path)

    # We'll use images as the source of truth. We assume a .txt file
    # exists for every .jpg file.

    # Check if source directory exists
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"Error: Source image directory not found at {SOURCE_IMAGES_DIR}")
        print("Please create it and move all your .jpg files there.")
        return

    # Get list of all images
    image_files = glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*.jpg"))
    if not image_files:
        # Try .png if .jpg is not found
        image_files = glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*.png"))

    if not image_files:
        print(f"Error: No .jpg or .png images found in {SOURCE_IMAGES_DIR}")
        return

    print(f"Found {len(image_files)} total images.")

    # Shuffle and split
    random.shuffle(image_files)

    split_index = int(len(image_files) * TRAIN_SPLIT)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Splitting into {len(train_files)} training files and {len(val_files)} validation files.")

    # Move files

    def copy_files(file_list, img_dest, lbl_dest):
        """Copies image and its corresponding label file."""
        copied_count = 0
        for img_path in file_list:

            # Get the base filename (e.g., "image1.jpg")
            img_filename = os.path.basename(img_path)

            # Get the base name without extension (e.g., "image1")
            base_filename = os.path.splitext(img_filename)[0]

            # Create the corresponding label file path (e.g., "all_labels/image1.txt")
            lbl_filename = base_filename + ".txt"
            lbl_path = os.path.join(SOURCE_LABELS_DIR, lbl_filename)

            # Check if label file exists
            if not os.path.exists(lbl_path):
                print(f"Warning: Label file not found for {img_filename}. Skipping this file.")
                continue

            # Define destination paths
            dest_img_path = os.path.join(img_dest, img_filename)
            dest_lbl_path = os.path.join(lbl_dest, lbl_filename)

            # Copy the files
            shutil.copy(img_path, dest_img_path)
            shutil.copy(lbl_path, dest_lbl_path)
            copied_count += 1

        return copied_count

    print("Copying training files...")
    train_count = copy_files(train_files, train_img_path, train_lbl_path)

    print("Copying validation files...")
    val_count = copy_files(val_files, val_img_path, val_lbl_path)

    print("\n--- Dataset Preparation Complete ---")
    print(f"Total images processed: {len(image_files)}")
    print(f"Successfully copied to train: {train_count} images/labels")
    print(f"Successfully copied to val: {val_count} images/labels")
    print("--------------------------------------")


if __name__ == "__main__":
    prepare_dataset()