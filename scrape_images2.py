
# Purpose: Download, clean, and resize 150 web images per class


from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os, hashlib

# -----------------------------
# Configuration
# -----------------------------
CLASSES = [
    "Keyboard", "Mouse", "Pen Pencil", "File Folder", "Bottle",
    "Mug", "Tape Dispenser", "Calculator", "Stapler", "Book"
]

BASE_DIR = "data/train"
MAX_DOWNLOADS_PER_CLASS = 150  
TARGET_SIZE = (640, 640)

# -----------------------------
# Helper Functions
# -----------------------------
def hash_file(path):
    """Return md5 hash of an image file for duplicate detection."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def clean_and_resize(folder):
    """Verify, resize to 640x640, and remove bad/duplicate images."""
    print(f"🧹 Cleaning folder: {folder}")
    hashes = set()
    valid_count = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            with Image.open(path) as img:
                img.verify()  # check corruption
            # reopen for resize
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
# Main Scraping Loop
# -----------------------------
os.makedirs(BASE_DIR, exist_ok=True)
print("🚀 Starting bulk image scraping (100+ images per class)...")

for class_name in CLASSES:
    query = f"office {class_name}"
    folder = os.path.join(BASE_DIR, class_name.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    print(f"\n🔍 Downloading images for: {class_name}")

    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={"root_dir": folder}
    )
    crawler.crawl(
        keyword=query,
        max_num=MAX_DOWNLOADS_PER_CLASS,
        filters={"size": "medium"}
    )

    # Clean & resize
    valid = clean_and_resize(folder)
    print(f"✅ {valid} valid images ready for: {class_name}")

print("\n🎉 Scraping complete!")
print("All images are resized to 640x640 and saved in:")
print(f"📁 {os.path.abspath(BASE_DIR)}")
print("Please manually review for relevance before training.")
