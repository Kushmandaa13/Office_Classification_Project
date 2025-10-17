# ================================================================
# scrape_more_images.py
# Purpose: Add more images to existing dataset folders
# ================================================================

from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os, hashlib

# -----------------------------
# CONFIGURATION
# -----------------------------
CLASSES = [
    "Keyboard", "Mouse", "Pen Pencil", "File Folder", "Bottle",
    "Mug", "Tape Dispenser", "Calculator", "Stapler", "Book"
]

BASE_DIR = "data/train"
ADDITIONAL_IMAGES_PER_CLASS = 50     # how many new images you want to add
TARGET_SIZE = (640, 640)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def hash_file(path):
    """Return md5 hash of an image file for duplicate detection."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def clean_and_resize(folder):
    """Verify, resize, and remove bad or duplicate images."""
    print(f"🧹 Cleaning folder: {folder}")
    hashes = set()
    valid_count = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            with Image.open(path) as img:
                img.verify()  # check corruption
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize(TARGET_SIZE)
                file_hash = hash_file(path)
                if file_hash in hashes:
                    os.remove(path)
                    continue
                hashes.add(file_hash)
                img.save(path, "JPEG", quality=90)
                valid_count += 1
        except Exception:
            os.remove(path)
    return valid_count

# -----------------------------
# MAIN SCRAPING LOOP
# -----------------------------
print("🚀 Starting incremental image scraping...")

for class_name in CLASSES:
    folder = os.path.join(BASE_DIR, class_name.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    
    existing = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    target = existing + ADDITIONAL_IMAGES_PER_CLASS
    
    print(f"\n🔍 {class_name}: currently {existing} images, target {target}")
    remaining = ADDITIONAL_IMAGES_PER_CLASS
    
    if remaining <= 0:
        print("Already at or above target. Skipping...")
        continue

    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={"root_dir": folder}
    )
    
    query = f"office {class_name}"
    crawler.crawl(
        keyword=query,
        max_num=remaining,
        filters={"size": "medium"}
    )

    valid = clean_and_resize(folder)
    print(f"✅ Folder '{class_name}' now has {valid} valid resized images.")

print("\n🎉 Incremental scraping complete!")
print("All new images have been added, cleaned, and resized to 640x640.")
