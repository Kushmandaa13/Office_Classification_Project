import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    # Get all class folders
    class_folders = [f for f in os.listdir(source_dir) 
                    if os.path.isdir(os.path.join(source_dir, f))]
    
    print(f"Found {len(class_folders)} classes: {class_folders}\n")
    
    for class_name in class_folders:
        source_class_path = os.path.join(source_dir, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(source_class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        total_images = len(image_files)
        print(f"Processing class '{class_name}': {total_images} images")
        
        # Shuffle images randomly
        random.shuffle(image_files)
        
        # Calculate split indices
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        # test_count gets the remainder to ensure all images are used
        
        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]
        
        print(f"  - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # Create class folders in validation and test directories
        val_class_path = os.path.join(val_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        # Move validation images
        for img in val_images:
            src = os.path.join(source_class_path, img)
            dst = os.path.join(val_class_path, img)
            shutil.move(src, dst)
        
        # Move test images
        for img in test_images:
            src = os.path.join(source_class_path, img)
            dst = os.path.join(test_class_path, img)
            shutil.move(src, dst)
        
        print(f"Completed '{class_name}'\n")
    
    print("Dataset split completed successfully!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths - adjust these to your actual paths
    base_dir = "data"
    train_folder = os.path.join(base_dir, "train")
    val_folder = os.path.join(base_dir, "validation")
    test_folder = os.path.join(base_dir, "test")
    
    # Create validation and test folders if they don't exist
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    print("=" * 50)
    print("Image Dataset Splitter (70/15/15)")
    print("=" * 50)
    print(f"Source: {train_folder}")
    print(f"Validation: {val_folder}")
    print(f"Test: {test_folder}")
    print("=" * 50 + "\n")
    
    # Perform the split
    split_dataset(
        source_dir=train_folder,
        train_dir=train_folder,
        val_dir=val_folder,
        test_dir=test_folder,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )