Office Items Image Dataset 

Course:
AI in Robotics (PDE 3802) — 2025–26  
Project: Office Item Classifier Robot  
Team: The Back-Propagator

Dataset Overview

The Office Items Dataset is a custom image dataset designed to train and evaluate a computer vision model that classifies and detects common office objects for use in an autonomous office-organizing robot.

This dataset includes 10 office item categories collected through a combination of:
 Manual photography (real office scenes)  
 Web scraping using the `icrawler` Python package  

Each image contains one clearly visible object belonging to one of the following classes:

| # | Class Name |
|:-:|:------------|
| 1 | Book |
| 2 | Bottle |
| 3 | Calculator |
| 4 | File Folder |
| 5 | Keyboard |
| 6 | Mouse |
| 7 | Mug |
| 8 | Pen Pencil |
| 9 | Stapler |
| 10 | Tape Dispenser |


Directory Structure

data/
├── train/ (70%)
├── val/ (15%)
└── test/ (15%)


Each subfolder (train, val, test) contains 10 class folders, one per object category.

Example:
data/train/
├── Book/
├── Bottle/
├── Calculator/
├── File_Folder/
├── Keyboard/
├── Mouse/
├── Mug/
├── Pen_Pencil/
├── Stapler/
└── Tape_Dispenser/


Data Summary 

| Split | Total Images | Approx. % of Total |
|:------|--------------:|------------------:|
| Train | 979 | ~70% |
| Val | 318 | ~22% |
| Test | 163 | ~11% |
| Total | 1,460 | 100% |

> Slight variations occur due to class-level rounding and incremental scraping.


Per-Class Summary 
| Class | Train | Val | Test | Total |
|:------|------:|----:|----:|------:|
| Book | 77 | 16 | 18 | 111 |
| Bottle | 75 | 16 | 17 | 108 |
| Calculator | 51 | 11 | 12 | 74 |
| File Folder | 61 | 13 | 14 | 88 |
| Keyboard | 73 | 15 | 17 | 105 |
| Mouse | 74 | 15 | 17 | 106 |
| Mug | 72 | 15 | 17 | 104 |
| Pen Pencil | 49 | 10 | 12 | 71 |
| Stapler | 79 | 17 | 18 | 114 |
| Tape Dispenser | 72 | 15 | 16 | 103 |
| Total | 683 | 143 | 156 | ~982 |


Data Collection Process

Sources
Primary: Self-captured photos using webcam and mobile phone in office settings  
Secondary: Web-scraped images via Google using `icrawler` Python library  

Search Queries
Each class was scraped using queries such as:
`"office Book"`, `"office Mug"`, `"desk Calculator"`, `"computer Keyboard"`, etc.

Filtering & Cleaning
 Duplicates removed using MD5 hashing
 Corrupted or grayscale files deleted
 Irrelevant or low-quality images manually removed
 All images resized to 640×640 px (RGB)
 Verified for visual clarity and realism

Preprocessing

All images are preprocessed for training:
 Resized: `640 × 640` pixels  
 Color space: `RGB`
 Normalization: `(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
 Augmentations (during training): random flip, rotation, brightness/contrast, crop  


Dataset Splitting

 70% → Training (used for model learning)  
 15% → Validation (used to tune hyperparameters)  
 15% → Testing (used for final evaluation)  

Splitting was automated using a Python script (`split_dataset_safe.py`).

Intended Use

This dataset supports:
 Image classification or object detection tasks  
 Research in robotic perception systems  
 Coursework and experimentation in AI in Robotics (PDE 3802)



Limitations

 Some web-scraped images may contain backgrounds or watermarks  
 Class imbalance is minimal but not eliminated  
 Images are 2D only — no depth, segmentation, or pose annotations  
 Lighting and viewpoint vary across samples  


Version History

| Version | Date | Description |
|:--------|:------|:------------|
| 1.0 | Oct 2025 | Initial dataset with 10 classes (~1,400 images) |
| 1.1 | Nov 2025 | Added extra images + improved validation/test balance |



References
- [PyTorch `ImageFolder` Documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)  
- [iCrawler Library](https://pypi.org/project/icrawler/)  
- [OpenCV Documentation](https://docs.opencv.org/)  
- [YOLOv8 Docs](https://docs.ultralytics.com)




