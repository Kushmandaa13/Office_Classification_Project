# ================================================================
# resplit_dataset_safe.py
# Purpose: Reorganize dataset into train/val/test (70/15/15)
# Safe for Windows / OneDrive
# ================================================================

import os, shutil, random

BASE_DIR = "data"
SRC_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def clean_folder(folder_path):
    """Safely clear contents of a folder (Windows/OneDrive-friendly)."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except Exception as e:
                print(f"⚠️ Could not remove file {f}: {e}")
        for d in dirs:
            try:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            except Exception as e:
                print(f"⚠️ Could not remove subfolder {d}: {e}")


# --- Clean existing val/test folders ---
print("🧹 Cleaning val/ and test/ directories...")
clean_folder(VAL_DIR)
clean_folder(TEST_DIR)

# --- Perform splitting ---
print("\n🚀 Starting dataset re-split (70% train, 15% val, 15% test)...")

for cls in sorted(os.listdir(SRC_DIR)):
    src_cls_path = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(src_cls_path):
        continue

    all_imgs = [f for f in os.listdir(src_cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_imgs)

    total = len(all_imgs)
    if total == 0:
        print(f"⚠️ Skipping empty class folder: {cls}")
        continue

    n_train = int(total * SPLITS["train"])
    n_val = int(total * SPLITS["val"])

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:]
    }

    print(f"\n📁 Class: {cls}")
    print(f"   Total: {total} → Train {len(splits['train'])}, "
          f"Val {len(splits['val'])}, Test {len(splits['test'])}")

    # Copy images into val/test
    for split in ["val", "test"]:
        dest_dir = os.path.join(BASE_DIR, split, cls)
        os.makedirs(dest_dir, exist_ok=True)
        for f in splits[split]:
            src = os.path.join(src_cls_path, f)
            dst = os.path.join(dest_dir, f)
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"⚠️ Could not copy {src} → {dst}: {e}")

print("\n✅ Dataset re-splitting complete!")
print("Your dataset is now under data/train, data/val, and data/test.")
