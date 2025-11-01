# OfficeObjects Classifier (ResNet-50)

This project, by **The Back Propagators**, implements an "OfficeObjects Classifier" using PyTorch. The system can identify 10 common office items with high accuracy and includes both a training pipeline and a real-time classification GUI application.

It includes two main components:
- **Jupyter Notebook** (`cw1_classifier.ipynb`): A notebook to train the ResNet-50 model, evaluate its performance, and save the final weights.
- **GUI Application** (`ui.py`): A Tkinter application that loads the trained model to perform live, real-time classification using a webcam or from an uploaded image.

## Features

- **Model**: ResNet-50 (using transfer learning)
- **Framework**: PyTorch
- **Training**: Full training and evaluation pipeline in the Jupyter Notebook
- **GUI**: A user-friendly Tkinter app for easy inference
- **Live Detection**: Classifies objects in real-time using a webcam
- **Image Upload**: Classifies static images from file system
- **Object Detection**: Uses OpenCV to find and highlight the object with a bounding box
- **Robustness**: Uses Test-Time Augmentation (TTA) in the UI for more reliable predictions
- **Color-Coded Output**: Green box = recognized (≥70% confidence), Red box = unknown (<70% confidence)

## 10 Target Classes

1. Book
2. Bottle
3. Calculator
4. File_Folder
5. Keyboard
6. Mouse
7. Mug
8. Pen_Pencil
9. Stapler
10. Tape_Dispenser

## Dataset: OfficeObjects-10

The model is trained on a custom dataset built for this task.

### Dataset Overview
- **Data Source**: A mix of custom images (~30% captured via webcam/mobile) and images from public datasets (~70% from Google Images, Open Images Dataset)
- **Total Images**: 3,206
- **Image Format**: `.jpg` or `.png`
- **Image Size (after augmentation)**: 224×224

### Data Split
- **Train**: 2,257 images (70%)
- **Validation**: 379 images (15%)
- **Test**: 570 images (15%)

### Data Augmentation
Applied during training:
- Random horizontal flip
- Random rotation (up to 15 degrees)
- Color jitter (brightness, contrast, saturation: ±0.2)
- Random resized crop
- Normalization using ImageNet statistics

## Model Performance

### Training Results
- **Best Validation Accuracy**: 89.97%
- **Training Epochs**: 25 (with early stopping at patience=6)
- **GPU Used**: NVIDIA GeForce MX330

### Test Set Evaluation
- **Test Accuracy**: 89.12%
- **Macro F1-Score**: 0.8860
- **Error Rate**: 1.05% (6 out of 570 images misclassified)

### Confusion Matrix
See `confusion_matrix.png` for detailed per-class performance visualization.

### Error Analysis

**Total Misclassified**: 6 / 570 images (1.05% error rate)

**Most Common Misclassifications**:
- Pen_Pencil → Mouse (1 case)
- Mug → Mouse (1 case)
- File_Folder → Book (1 case)
- Mouse → Keyboard (1 case)
- File_Folder → Keyboard (1 case)

**Root Causes**:
- Similar shape or color between classes (e.g., Mug and Bottle, Mouse and small objects)
- Partial occlusion in images
- Varying lighting conditions
- Different viewing angles

**Mitigation Strategies**:
- Extended data augmentation with more rotation and brightness variations
- Increased dataset size with more diverse backgrounds
- Improved training data collection with consistent lighting

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone the Repository

git clone https://github.com/Kushmandaa13/Office_Classification_Project.git
cd Office_Classification_Project

### Step 2: Create a Conda Environment (Recommended)

conda create --name office_classifier_env python=3.11
conda activate office_classifier_env

### Step 3: Install Required Libraries

Install all dependencies for both the training notebook and the GUI:

pip install torch torchvision numpy matplotlib scikit-learn pillow opencv-python

**GPU Support (Optional)**: If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For other GPU types or detailed instructions, visit PyTorch's official website (https://pytorch.org/get-started/locally/).

### Step 4: Dataset Setup

Place your dataset in the project root directory following this structure:

./
├── cw1_classifier.ipynb
├── ui.py
├── best_model_weights.pth
├── data/
│   ├── train/
│   │   ├── Book/
│   │   ├── Bottle/
│   │   ├── Calculator/
│   │   ├── File_Folder/
│   │   ├── Keyboard/
│   │   ├── Mouse/
│   │   ├── Mug/
│   │   ├── Pen_Pencil/
│   │   ├── Stapler/
│   │   └── Tape_Dispenser/
│   ├── val/
│   │   ├── Book/
│   │   └── ... (all 10 classes)
│   └── test/
│       ├── Book/
│       └── ... (all 10 classes)
├── README.md
└── training_history.png

## How to Use

This is a 2-step process: first, you must train the model, then you can run the UI.

### Step 1: Train the Model (Required First)

Before you can run the UI, you must train the model to generate the best_model_weights.pth file.

1. Activate your Conda environment:
   conda activate office_classifier_env

2. Launch Jupyter Notebook:
   jupyter notebook

3. Open cw1_classifier.ipynb and run all cells sequentially from top to bottom:
   - Data loading and augmentation
   - Model setup (ResNet-50 with fine-tuning)
   - Training with early stopping
   - Evaluation on test set
   - Error analysis visualization

4. Wait for training to complete (~15-30 minutes depending on GPU). The notebook will:
   - Save best_model_weights.pth in your project root
   - Generate training_history.png (training curves)
   - Generate confusion_matrix.png (per-class performance)
   - Generate error_analysis.png (misclassified examples)
   - Save evaluation_metrics.txt (detailed metrics)

### Step 2: Run the GUI Application

Once best_model_weights.pth exists, you can run the real-time classifier.

1. In your terminal (with the Conda environment still active), run:
   python ui.py

2. The GUI window will open. You can now:
   - Click "Upload Image" to select a static image file for classification
   - Click "Live Camera" to start your webcam for real-time detection
   - View the predicted class, confidence score, and detection status

## Expected Outputs

### Training Phase (cw1_classifier.ipynb)

Console Output:
- Loss and accuracy per epoch for both training and validation sets
- Early stopping notification when validation accuracy plateaus
- Best validation accuracy achieved
- Final test set metrics

Generated Files:
- best_model_weights.pth — Trained model weights (use with ui.py)
- training_history.png — Line plot showing training/validation loss and accuracy over epochs
- confusion_matrix.png — Heatmap showing per-class classification performance
- error_analysis.png — Grid of 6 misclassified examples with true/predicted labels
- evaluation_metrics.txt — Detailed classification report with precision, recall, F1-score per class

### GUI Application (ui.py)

When uploading an image:
- Displays the image with a bounding box around the detected object
- Shows predicted class (e.g., "Mug")
- Shows confidence score (e.g., "99.21%")
- Shows detection status: DETECTED (green, ≥70% confidence) or UNKNOWN (red, <70% confidence)

When using live camera:
- Real-time video stream (~30 FPS) with continuous object detection
- Bounding box and predictions update in real-time
- Color-coded output: Green = recognized, Red = unknown
- Smooth camera feed using threading

Confidence Threshold:
- Objects with ≥70% confidence are marked as "DETECTED" (green box)
- Objects with <70% confidence are marked as "UNKNOWN" (red box)

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'torch'
**Solution**: 
- Ensure your Conda environment is activated: conda activate office_classifier_env
- Reinstall PyTorch: pip install torch torchvision

### Issue: FileNotFoundError: best_model_weights.pth not found
**Solution**: 
- You must run the Jupyter Notebook first to train and save the model
- Ensure the notebook completes all training epochs without errors
- Check that best_model_weights.pth exists in your project root directory

### Issue: Camera not accessible or Could not access camera
**Solution**: 
- Check that no other application is using the webcam
- On Windows: Check camera permissions in Settings → Privacy → Camera
- On macOS: Grant camera access to Terminal/Python in System Preferences → Security & Privacy
- Try restarting the application
- If using a USB camera, ensure it's properly connected

### Issue: Slow inference on CPU or UI is laggy
**Solution**: 
- Consider installing GPU-enabled PyTorch (see Setup section)
- Reduce frame rate by adjusting code: change dataloaders batch size or frame skip rate
- Close other applications consuming resources
- Note: CPU inference will be 10-50x slower than GPU

### Issue: CUDA out of memory during training
**Solution**: 
- Reduce batch size in notebook: change batch_size = 16 to batch_size = 8
- Reduce number of workers: change num_workers=2 to num_workers=0
- Close other GPU applications

### Issue: Model predicts everything as Unknown
**Solution**: 
- Check that best_model_weights.pth was properly trained (not default weights)
- Verify the confidence threshold in ui.py: self.CONFIDENCE_THRESHOLD = 0.70
- Ensure good lighting when using the camera
- Try uploading a clear test image from the dataset

### Issue: No module named 'tkinter'
**Solution**: 
- Tkinter usually comes with Python, but on Linux you may need: sudo apt-get install python3-tk
- Ensure you're using the correct Python version with Tkinter installed

### Issue: Out of memory when loading dataset
**Solution**: 
- Ensure dataset directory structure is correct (see Dataset Setup section)
- Check disk space availability
- Reduce number of workers: num_workers=0 in the notebook

## Project Structure

Office_Classification_Project/
│
├── cw1_classifier.ipynb          # Jupyter notebook for training & evaluation
├── ui.py                          # Tkinter GUI application
├── best_model_weights.pth         # Trained model weights (generated after training)
├── README.md                      # This file
│
├── data/                          # Dataset directory
│   ├── train/                     # Training images (70%)
│   ├── val/                       # Validation images (15%)
│   └── test/                      # Test images (15%)
│
├── training_history.png           # Training curves (generated)
├── confusion_matrix.png           # Confusion matrix heatmap (generated)
├── error_analysis.png             # Misclassified examples (generated)
└── evaluation_metrics.txt         # Detailed metrics report (generated)

## System Concept Video (3 minutes)

Watch the system design overview covering:
- Problem statement and use case
- Robot mission and tasks
- Hardware components (sensors, actuators, compute)
- Software architecture (ROS 2, perception, planning)
- Dataset and model training approach


## Code Walkthrough Video (2 minutes)

Watch the code demonstration showing:
- Training pipeline in Jupyter Notebook
- Model evaluation and performance metrics
- GUI application in action
- Real-time classification demo with webcam and uploaded images


## Model Architecture

**Base Model**: ResNet-50 (pretrained on ImageNet)

**Fine-tuning Strategy**:
- Freeze layers: layer1, layer2
- Fine-tune: layer3, layer4, and fully connected layer
- Added dropout (0.4) before classification head

**Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)

**Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)

**Loss Function**: Cross-Entropy Loss

**Early Stopping**: Patience=6 epochs (stops if validation accuracy doesn't improve)

## Dataset Card

| Property | Value |
|----------|-------|
| Dataset Name | OfficeObjects-10 |
| Task | Image Classification |
| Number of Classes | 10 |
| Total Images | 3,206 |
| Train/Val/Test Split | 70% / 15% / 15% |
| Data Source | ~30% custom captured, ~70% public datasets |
| Image Format | JPEG / PNG |
| Image Dimensions | 224×224 (after augmentation) |
| Augmentation | Flip, rotation, brightness, contrast, crop |
| License | For educational purposes |

## Authors

**Group Name**: The Back Propagators

**Members**:
- Kushmandaa Devi Bhaugeeruth (M00965269) - Model training, UI development, Required Skills & Hardware sections
- Leynah Reeya Veerapen (M00976615) - Software Architecture, Dataset & Model Plan sections, coding support
- Kimberly Alexya Ramasamy (M00957365) - Introduction, Mission & Tasks, Risk & Safety, Budget/BOM sections, report formatting

## References

1. R. R. Murphy, Introduction to AI Robotics, 2nd ed. MIT Press, 2019.
2. F. X. Govers, Artificial Intelligence for Robotics. Packt Publishing, 2018.
3. S. Marsland, Machine Learning: An Algorithmic Perspective, 2nd ed. CRC Press, 2015.
4. G. A. Bekey, Autonomous Robots: From biological inspiration to implementation and control. MIT Press, 2005.
5. Khan, S.M.A. et al., "Software Architecture in AI Enabled Systems: A Systematic Literature Review," IEEE, 2023.
6. ROS 2 Documentation, "Robot Operating System 2 Overview and Architecture," 2024.
7. Cavalcanti, A. et al. (eds.), Software Engineering for Robotics. Springer Nature, 2021.
8. Quigley, M. et al., "ROS: An open-source Robot Operating System," ICRA, 2009.
9. Siciliano, B., Khatib, O., Springer Handbook of Robotics. Springer, 2016.
10. Huang, A. et al., "Robotic Perception in Unstructured Environments: A Review," Robotics and Autonomous Systems, 2021.
11. Redmon, J., Farhadi, A., "YOLOv3: An Incremental Improvement," arXiv:1804.02767, 2018.
12. Howard, A. et al., "Searching for MobileNetV3," IEEE/CVF ICCV, 2019.
13. Andras, I. et al., "Artificial intelligence and robotics: a combination that is changing the operating room," World J Urol, 38:2359–2366, 2020.
14. Nagatani, K. et al., "Innovative technologies for infrastructure construction and maintenance through collaborative robots," Advanced Robotics, 35(11):715-722, 2021.
15. Niloy, M.A. et al., "Critical design and control issues of indoor autonomous mobile robots: A review," IEEE Access, 9:35338-35370, 2021.
16. Neupane, S. et al., "Security considerations in ai-robotics: A survey of current methods, challenges, and opportunities," IEEE Access, 12:22072-22097, 2024.
17. Thomasen, K., "Safety in Artificial Intelligence & Robotics in Canada," Can. B. Rev., 101:61, 2023.
18. Hendrycks, D., Introduction to AI safety, ethics, and society. Taylor & Francis, 2025.

## License

This project is for educational purposes as part of the PDE3802 (AI in Robotics) module at Middlesex University.

## Contact

For questions or issues, please reach out to the team or contact the module tutor.

---

**Last Updated**: 1 November 2025
**Repository**: https://github.com/Kushmandaa13/Office_Classification_Project

