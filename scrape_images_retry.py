# ======================================================================
# TARGETED WEB IMAGE SCRAPER FOR MISSING DATA
# ======================================================================
# This script specifically targets the classes that had poor download rates 
# (Mouse and Book) to recover the necessary volume for the training set.
# Run this script, then complete the manual collection for all 10 classes.

from icrawler.builtin import GoogleImageCrawler
import os

# --- Configuration ---
# Only target the classes with missing data
CLASSES_TO_RETRY = [
    "Computer Mouse (office scroll clicker)", 
    "Desk Book Hardcover Paperback" 
]

# The target directory (data/train)
BASE_DIR = 'data/train'

# Higher volume to ensure we hit the minimum threshold
MAX_DOWNLOADS_PER_CLASS = 100 

# --- Execution ---
print("Starting targeted image scraping for Mouse and Book...")

for class_query in CLASSES_TO_RETRY:
    # Use the first word as the folder name (e.g., 'Computer' -> Mouse folder, 'Desk' -> Book folder)
    if 'Mouse' in class_query:
        class_name = "Mouse"
    elif 'Book' in class_query:
        class_name = "Book"
    else:
        # Fallback for unexpected class names
        class_name = class_query.split(' ')[0] 
        
    # Define the output path for the specific folder
    output_path = os.path.join(BASE_DIR, class_name)
    
    # 3. Create the folder if it doesn't exist (should exist)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created missing directory: {output_path}")

    # 4. Initialize the crawler
    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={'root_dir': output_path}
    )

    print(f"\n--- Downloading {MAX_DOWNLOADS_PER_CLASS} images for: {class_name} (Search Query: {class_query}) ---")
    
    # 5. Start the crawl with specific query terms
    crawler.crawl(
        keyword=class_query,
        max_num=MAX_DOWNLOADS_PER_CLASS,
        filters={'size': 'medium'}
    )

print("\n\n---------------------------------")
print("TARGETED SCRAPING COMPLETE!")
print("The next critical step is manual collection for realism.")
print("---------------------------------")