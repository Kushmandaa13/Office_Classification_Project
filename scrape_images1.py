# WEB IMAGE SCRAPER FOR TRAINING DATA

from icrawler.builtin import GoogleImageCrawler
import os

# --- Configuration ---
# 10 defined office item classes
CLASSES = [
    "Mug", "Bottle", "Book", "Stapler", "Keyboard", 
    "Mouse", "Calculator", "Tape Dispenser", "Pen Pencil", "File Folder"
]

# The target directory where the data is stored
# It automatically creates/uses folders inside data/train/
BASE_DIR = 'data/train'

# Target number of images to download per class from the web.
MAX_DOWNLOADS_PER_CLASS = 70 

# --- Execution ---
print("Starting bulk image scraping. This may take several minutes...")

for class_name in CLASSES:
    # 1. Define the search query
    query = f"office supply {class_name}"
    
    # 2. Define the output path, replacing spaces with underscores for compatibility
    output_path = os.path.join(BASE_DIR, class_name.replace(' ', '_'))
    
    # 3. Create the folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 4. Initialize the crawler
    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={'root_dir': output_path}
    )

    print(f"\n--- Downloading {MAX_DOWNLOADS_PER_CLASS} images for: {class_name} ---")
    
    # 5. Start the crawl
    crawler.crawl(
        keyword=query,
        max_num=MAX_DOWNLOADS_PER_CLASS,
        filters={'size': 'medium'}
    )

print("\n\n---------------------------------")
print("SCRAPING COMPLETE!")
print(f"Web images have been saved to the {BASE_DIR} folders.")
print("The next step is to manually add your own realistic images.")
print("---------------------------------")