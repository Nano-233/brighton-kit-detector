"""
TRAINING SCRIPT FOR FOOTBALL KIT DETECTOR

This script trains a YOLO model and then automatically organizes the
output files into the project's 'models/' and 'training_analytics/' directories.
"""

import os
import shutil
from ultralytics import YOLO

# --- CONFIGURATION ---
DATA_CONFIG = 'dataset/data.yaml'
MODEL_NAME = 'yolo11s.pt'
EPOCHS = 60
IMAGE_SIZE = 640

# --- DESTINATION PATHS ---
PROJECT_NAME = 'runs/detect'
RUN_NAME = 'train'
ANALYTICS_DEST_DIR = 'training_analytics'
MODELS_DEST_DIR = 'models'
BEST_MODEL_NAME = 'best.pt' # The name of the model you want to save

def main():
    """
    Runs the complete training and file organization pipeline.
    """
    print(f"--- Initializing Training ---")
    print(f"Model: {MODEL_NAME}, Epochs: {EPOCHS}, ImgSize: {IMAGE_SIZE}")
    
    model = YOLO(MODEL_NAME) 

    # The 'train' method handles all training, validation, and GPU detection.
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME
    )
    
    print("\n--- Training Complete ---")

    # --- 2. POST-TRAINING FILE ORGANIZATION ---
    print("Organizing output files...")

    # Get the path to the training run directory (e.g., 'runs/detect/train')
    source_dir = results.save_dir

    # Create destination directories if they don't exist
    os.makedirs(ANALYTICS_DEST_DIR, exist_ok=True)
    os.makedirs(MODELS_DEST_DIR, exist_ok=True)

    source_model_path = os.path.join(source_dir, 'weights', BEST_MODEL_NAME)
    dest_model_path = os.path.join(MODELS_DEST_DIR, BEST_MODEL_NAME)

    if os.path.exists(source_model_path):
        print(f"Copying model to {dest_model_path}")
        shutil.copy(source_model_path, dest_model_path)
    else:
        print(f"Warning: Could not find {BEST_MODEL_NAME} at {source_model_path}")

    # We copy everything from the run folder EXCEPT the 'weights' directory
    print(f"Copying analytics files to {ANALYTICS_DEST_DIR}")
    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)
        dest_item_path = os.path.join(ANALYTICS_DEST_DIR, item)

        if item == 'weights':
            continue  # Skip the weights folder

        if os.path.isfile(source_item_path):
            shutil.copy(source_item_path, dest_item_path)
        elif os.path.isdir(source_item_path):
            shutil.copytree(source_item_path, dest_item_path, dirs_exist_ok=True)

    print("\n--- File Organization Complete ---")
    print(f"Best model saved to: {dest_model_path}")
    print(f"Training analytics saved to: {ANALYTICS_DEST_DIR}")

if __name__ == "__main__":
    main()