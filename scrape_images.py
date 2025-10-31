import os
import requests
import time
from bing_image_downloader import downloader
from pathlib import Path

def download_images_bing(class_names, output_dir, images_per_class=100):
    
    print("=" * 70)
    print("IMAGE SCRAPER - Downloading from Bing")
    print("=" * 70)
    print(f"Target: {images_per_class} images per class")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()
    
    for idx, class_name in enumerate(class_names, 1):
        print(f"[{idx}/{len(class_names)}] Downloading images for: {class_name}")
        
        try:
            # Bing image downloader
            downloader.download(
                query=class_name,
                limit=images_per_class,
                output_dir=output_dir,
                adult_filter_off=True,
                force_replace=False,
                timeout=15,
                verbose=False
            )
            
            print(f"Completed: {class_name}")
            
        except Exception as e:
            print(f"Error downloading {class_name}: {str(e)}")
        
        print()
        time.sleep(1)  # Be polite to the server
    
    print("=" * 70)
    print("Download completed!")
    print("=" * 70)

def organize_downloaded_images(temp_dir, train_dir, class_names):
    
    print("\nOrganizing images into train folder...")
    
    for class_name in class_names:
        # Bing downloader creates folders with class names
        source_folder = os.path.join(temp_dir, class_name)
        target_folder = os.path.join(train_dir, class_name)
        
        if not os.path.exists(source_folder):
            print(f"Folder not found: {source_folder}")
            continue
        
        # Create target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)
        
        # Move images to train folder
        image_files = [f for f in os.listdir(source_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        count = 0
        for img_file in image_files:
            source_path = os.path.join(source_folder, img_file)
            
            # Create unique filename to avoid overwriting
            base_name, ext = os.path.splitext(img_file)
            target_path = os.path.join(target_folder, f"web_{base_name}{ext}")
            
            # If file exists, add counter
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(target_folder, f"web_{base_name}_{counter}{ext}")
                counter += 1
            
            # Move file
            os.rename(source_path, target_path)
            count += 1
        
        print(f"  ✓ {class_name}: {count} images added")
        
        # Remove empty source folder
        try:
            os.rmdir(source_folder)
        except:
            pass
    
    # Remove temp directory if empty
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    print("\nOrganization completed!")

if __name__ == "__main__":
    # Define your classes
    classes = [
        "Book",
        "Bottle",
        "Calculator",
        "File Folder",
        "Keyboard",
        "Mouse",
        "Mug",
        "Pen Pencil",
        "Stapler",
        "Tape Dispenser"
    ]
    
    # Define paths
    base_dir = "data"
    train_folder = os.path.join(base_dir, "train")
    temp_download_folder = "temp_downloads"
    
    # Number of images to download per class
    images_per_class = 100
    
    print("\n" + "=" * 70)
    print("IMPORTANT: Make sure you have installed the required package:")
    print("pip install bing-image-downloader")
    print("=" * 70)
    
    # Check if package is installed
    try:
        import bing_image_downloader
    except ImportError:
        print("\nERROR: bing-image-downloader is not installed!")
        print("Please run: pip install bing-image-downloader")
        exit(1)
    
    # Create temp download folder
    os.makedirs(temp_download_folder, exist_ok=True)
    
    # Download images
    download_images_bing(classes, temp_download_folder, images_per_class)
    
    # Organize images into train folder
    organize_downloaded_images(temp_download_folder, train_folder, classes)
    
    print("\nAll done! Check your train folder for the new images.")