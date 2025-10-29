OfficeObjects Classifier (ResNet-50)

This project, by The Back Propagators, implements an "OfficeObjects Classifier" using PyTorch. The system can identify 10 common office items.

It includes two main components:

Jupyter Notebook (cw1_classifier.ipynb): A notebook to train the ResNet-50 model, evaluate its performance, and save the final weights.

GUI Application (ui.py): A Tkinter application that loads the trained model to perform live, real-time classification using a webcam or from an uploaded image.

Features

Model: ResNet-50 (using transfer learning)

Framework: PyTorch

Training: Full training and evaluation pipeline in the Jupyter Notebook.

GUI: A user-friendly Tkinter app for easy inference.

Live Detection: Classifies objects in real-time using a webcam.

Image Upload: Classifies static images.

Detection: Uses OpenCV to find the largest object contour and draw a bounding box.

Robustness: Uses Test-Time Augmentation (TTA) in the UI for more reliable predictions.

10 Target Classes
Book
Bottle
Calculator
File_Folder
Keyboard
Mouse
Mug
Pen_Pencil
Stapler
Tape_Dispenser

Dataset: OfficeObjects-10

The model is trained on a custom dataset built for this task.

Data Source: A mix of custom images (~30% captured via webcam/mobile) and images from public datasets (~70%).

Total Images: 3,206

Data Split:

Train: 2,257 images (70%)

Validation: 379 images (15%)

Test: 570 images (15%)

Setup & Installation

To run this project, you'll need Python 3.x and the following libraries.

Clone the repository:

git clone [https://github.com/Kushmandaa13/Office_Classification_Project.git](https://github.com/Kushmandaa13/Office_Classification_Project.git)
cd Office_Classification_Project


Create a Conda environment (recommended):

conda create --name office_classifier_env python=3.11
conda activate office_classifier_env


Install the required libraries:
This command installs all dependencies for both the training notebook and the GUI.

pip install torch torchvision numpy matplotlib scikit-learn pillow opencv-python
(If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch by following the instructions on the official PyTorch website).

Data Setup:
Place your dataset in the root directory following this structure:

./
├── cw1_classifier.ipynb
├── ui.py
├── data/
│   ├── train/
│   │   ├── Book/
│   │   ├── Bottle/
│   │   └── ... (all 10 classes)
│   ├── val/
│   │   ├── Book/
│   │   └── ...
│   └── test/
│       ├── Book/
│       └── ...
└── README.md


How to Use

This is a 2-step process: first, you must train the model, then you can run the UI.

Step 1: Train the Model (Required First)

Before you can run the UI, you must train the model to generate the best_model_weights.pth file.

Make sure your Conda environment is active:

conda activate office_classifier_env


Launch Jupyter Notebook:

jupyter notebook


Open cw1_classifier.ipynb and run all cells sequentially from top to bottom.

This will train the model and save the file best_model_weights.pth in your project's root directory.

Step 2: Run the GUI Application

Once the best_model_weights.pth file exists, you can run the real-time classifier.

In your terminal (with the Conda environment still active), run the ui.py script:

python ui.py


The application will open. You can now:

Click "Upload Image" to select a static image file for classification.

Click "Live Camera" to start your webcam for real-time detection.

Authors

The Back Propagators
