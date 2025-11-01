"""
Office Object Classifier with Real-time Detection

This application uses ResNet-50 deep learning model to classify office objects
and detect them in images or video streams from a webcam.

Features:
- Upload and classify static images
- Real-time object detection via webcam
- Test-Time Augmentation (TTA) for improved accuracy
- Visual bounding boxes with confidence scores
- Color-coded results (Green for recognized, Red for unknown)
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from threading import Thread
import time


# CLASS: OfficeObjectClassifier (Main Program
"""
    Main application class that handles the GUI and object classification workflow.
    
    Responsibilities:
    - Initialize the Tkinter GUI interface
    - Load and manage the ResNet-50 deep learning model
    - Process images from file uploads or camera feed
    - Display results with bounding boxes and confidence scores
"""
class OfficeObjectClassifier:
    def __init__(self, root):
        #Initialize the application with GUI setup and model loading.
        self.root = root
        self.root.title("Office Object Classifier with Detection")
        self.root.geometry("1100x900")
        self.root.state('zoomed')
        self.root.configure(bg='#f0f4f8')

        # Device and Model Settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = [
            'Book', 'Bottle', 'Calculator', 'File_Folder', 'Keyboard',
            'Mouse', 'Mug', 'Pen_Pencil', 'Stapler', 'Tape_Dispenser'
        ]
        self.INPUT_SIZE = 224 # ResNet-50 standard input size
        self.CONFIDENCE_THRESHOLD = 0.70   
        
        # Load the pre-trained model
        self.model = self.load_model()
        
        # Define Test-Time Augmentation (TTA) transforms
        #Apply different transformations to the image for more robust predictions
        # This helps improve accuracy by averaging predictions from different augmented versions
        
        self.tta_transforms = [
        # Transform 1: Standard Preprocessing (Center Crop)
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        # Transform 2: Horizontal Flip for invariance
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
        # Transform 3: Small Rotation for robustness
                transforms.Resize(256),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        ]
        
        #Camera and UI state variables
        self.camera = None  # OpenCV video capture object
        self.camera_active = False  # Flag to track if camera is currently running
        self.current_frame = None  # Store the latest frame from camera
        self.screenshot_count = 0  # Counter for saved screenshots
        
        # Create GUI components
        self.create_widgets()

    def load_model(self):
        """
        Load the ResNet-50 model with custom classification head.
        
        This function attempts to load pre-trained weights from a saved checkpoint.
        If the checkpoint doesn't exist, it falls back to default ImageNet weights
        and initializes the model for transfer learning.
        
        Returns:
            torch.nn.Module: The loaded model in evaluation mode
        """
        try:
            # Create base ResNet-50 architecture (without pre-trained weights initially)
            model = models.resnet50(weights=None)
            
            # Get the number of input features to the final fully connected layer
            num_ftrs = model.fc.in_features
            
            # Replace the final layer with a custom classification head
            # Includes dropout for regularization and output layer for our 10 classes
            model.fc = nn.Sequential(
                nn.Dropout(0.4),  # 40% dropout to prevent overfitting
                nn.Linear(num_ftrs, len(self.class_names))  # Output 10 class logits
            )
            
            # Load custom trained weights from checkpoint file
            model.load_state_dict(torch.load('best_model_weights.pth', 
                                             map_location=self.device, 
                                             weights_only=True))
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode (disables dropout)
            print(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            # If checkpoint loading fails, use default ImageNet pre-trained weights
            print(f"Error loading model: {e}")
            print("Using default pretrained weights for demo...")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_ftrs, len(self.class_names))
            )
            model = model.to(self.device)
            model.eval()
            return model

    def create_widgets(self):
        """
        Create and layout all GUI components.
        
        Creates:
        - Title banner at the top
        - Upload Image and Live Camera buttons
        - Image display area
        - Detection results display (predicted class, confidence, status)
        """
        # Title section
        # Dark header with application title and description
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        tk.Label(title_frame, text="Office Object Classifier & Detector",
                 font=('Helvetica', 24, 'bold'), bg='#2c3e50', fg='white').pack(expand=True)
        tk.Label(title_frame, text="ResNet-50 | Real-time Detection | Office Objects",
                 font=('Helvetica', 10), bg='#2c3e50', fg='#ecf0f1').pack()

        # Main content frame
        content_frame = tk.Frame(self.root, bg='#f0f4f8')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Button Control Panel
        button_frame = tk.Frame(content_frame, bg='#f0f4f8')
        button_frame.pack(pady=10, anchor='center')
        
        #Upload Image Button
        self.upload_btn = tk.Button(button_frame, text="Upload Image",
                                    font=('Helvetica', 14, 'bold'),
                                    bg='#3498db', fg='white',
                                    width=15, height=2, cursor='hand2',
                                    command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=10)
        
        #Live Camera Button
        self.camera_btn = tk.Button(button_frame, text="Live Camera",
                                    font=('Helvetica', 14, 'bold'),
                                    bg='#2ecc71', fg='white',
                                    width=15, height=2, cursor='hand2',
                                    command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=1, padx=10)

        # Image Display Area
        display_wrapper = tk.Frame(content_frame, bg='#f0f4f8')
        display_wrapper.pack(pady=15, anchor='center')
        
        #  White Frame to hold the displayed image
        self.display_frame = tk.Frame(display_wrapper, bg='white', relief='solid', borderwidth=2, height=350, width=700)
        self.display_frame.pack(pady=15)
        self.display_frame.pack_propagate(False) #Maintain fixed size
        
        # Label to display the image or placeholder text
        self.image_label = tk.Label(self.display_frame, text="No image loaded",
                                    font=('Helvetica', 14), bg='white', fg='gray')
        self.image_label.pack(expand=True)
        
        # Instruction Label explaining bounding box colors
        tk.Label(content_frame, text="Green = Recognized | Red = Unknown (<70%)",
                 font=('Helvetica', 9), bg='#f0f4f8', fg='#555').pack(pady=5)

        # Results display area
        result_wrapper = tk.Frame(content_frame, bg='#f0f4f8')
        result_wrapper.pack(pady=10, anchor='center')
        
        self.result_frame = tk.Frame(result_wrapper, bg='#ecf0f1', relief='solid', borderwidth=2, width=700)
        self.result_frame.pack(fill='x')
        
        # Results title
        tk.Label(self.result_frame, text="Detection Result",
                 font=('Helvetica', 12, 'bold'), bg='#ecf0f1').pack(pady=5)
        
        # Result labels for predicted class, confidence, and status(Detected or Unknown)
        self.result_label = tk.Label(self.result_frame, text="Predicted: -",
                                     font=('Helvetica', 11), bg='#ecf0f1')
        self.result_label.pack(pady=3)
        
        self.confidence_label = tk.Label(self.result_frame, text="Confidence: -",
                                         font=('Helvetica', 11), bg='#ecf0f1')
        self.confidence_label.pack(pady=3)
        
        self.status_label = tk.Label(self.result_frame, text="Status: -",
                                     font=('Helvetica', 11, 'bold'), bg='#ecf0f1')
        self.status_label.pack(pady=3)

    def detect_object_region(self, image):
        """
        Detect the main object in the image and return its bounding box.
        
        Uses edge detection and contour analysis to find the largest object
        in the image. Falls back to a default center-based bounding box
        if no suitable object is found.
        
        Args:
            image (PIL.Image or np.ndarray): Input image to analyze
            
        Returns:
            tuple: Bounding box coordinates (x1, y1, x2, y2)
        """
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect edges using Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        # Find contours (object boundaries) in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found, get the largest one
        if contours:
            # Find the contour with the maximum area
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest) # Get Bounding Rectangle
            h_img, w_img = img_array.shape[:2]
            area = w * h
            img_area = h_img * w_img
            
            # Check if object size is reasonable (between 5% and 95% of image)
            if 0.05 < (area / img_area) < 0.95:
                # Add padding to the bounding box
                pad = 20
                x, y = max(0, x - pad), max(0, y - pad)
                w, h = min(w_img - x, w + 2 * pad), min(h_img - y, h + 2 * pad)
                return (x, y, x + w, y + h)
        
        # Fallback: Use center region if no suitable object detected
        # This creates a 20% margin from all edges
        h, w = img_array.shape[:2]
        m = int(min(h, w) * 0.1)
        return (m, m, w - m, h - m)

    def draw_detection_box(self, image, class_name, confidence, bbox):
        """
        Draw bounding box and label on the image.
        
        Args:
            image (PIL.Image): Image to annotate
            class_name (str): Predicted class name
            confidence (float): Confidence score (0-100)
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            tuple: (annotated_image, status_string)
                - annotated_image: PIL Image with drawn annotations
                - status_string: "DETECTED" or "UNKNOWN"
        """
        draw = ImageDraw.Draw(image)
        
        # Determine if object is recognized based on confidence threshold
        is_recognized = confidence >= self.CONFIDENCE_THRESHOLD * 100
         # Green box for recognized objects, red for unknown
        color = (0, 255, 0) if is_recognized else (255, 0, 0)
        # Create label text
        label = f"{class_name.replace('_', ' ')}: {confidence:.1f}%" if is_recognized else f"Unknown: {confidence:.1f}%"
        status = "DETECTED" if is_recognized else "UNKNOWN"
        # Draw bounding box rectangle
        draw.rectangle(bbox, outline=color, width=4)
        
        # Label Text Rendering
        try:
            # Try to use Arial font for better appearance
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default font if Arial not available
            font = ImageFont.load_default()

        # Get text bounding box to position background rectangle
        tb = draw.textbbox((0, 0), label, font=font)
        
        # Draw colored background rectangle behind text
        draw.rectangle([bbox[0], bbox[1] - tb[3] - 10, bbox[0] + tb[2] + 10, bbox[1]], 
                       fill=color)
        
        # Draw text label
        draw.text((bbox[0] + 5, bbox[1] - tb[3] - 5), label, 
                  fill=(255, 255, 255), font=font)
        
        return image, status

    def classify_image(self, image):
        """
        Classify an image using the ResNet-50 model with Test-Time Augmentation.
        
        Uses TTA to improve prediction robustness by averaging predictions
        from multiple augmented versions of the input image.
        
        Args:
            image (PIL.Image): Input image to classify
            
        Returns:
            tuple: (predicted_class_name, confidence_percentage)
        """
        try:
            self.model.eval()  # Ensure model is in evaluation mode
            all_probs = []

            # Test-Time Augmentation Loop
            # Run inference on each augmented version of the image
            with torch.no_grad():  # Disable gradient computation for faster inference
                for transform in self.tta_transforms:
                    # Apply augmentation and convert to tensor
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Get model predictions
                    outputs = self.model(image_tensor)
                    
                    # Convert logits to probabilities
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    all_probs.append(probs)

            # Average predictions across all augmented versions
            avg_probs = torch.mean(torch.cat(all_probs, dim=0), dim=0)
            
            # Find the class with highest average probability
            confidence, predicted_index = torch.max(avg_probs, 0)
            
            # Get the class name and convert confidence to percentage
            predicted_class = self.class_names[predicted_index.item()]
            confidence_percent = confidence.item() * 100

            return predicted_class, confidence_percent
            
        except Exception as e:
           print(f"Classification error: {e}")
           return "Error", 0.0


    def toggle_camera(self):
        """
        Toggle the camera on/off based on current state.
        
        If camera is active, stop it. If inactive, start it.
        """
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        """
        Initialize webcam capture and start the camera feed update thread.
        
        Attempts to open the default camera (index 0) and creates a daemon
        thread to continuously process camera frames.
        """

        try:
            # Open the default camera (index 0 is usually the built-in webcam)
            self.camera = cv2.VideoCapture(0)
            
            # Check if camera was opened successfully
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return
            
            self.camera_active = True
            
            # Update button appearance to indicate camera is running
            self.camera_btn.config(text="Stop Camera", bg='#e74c3c')
            
            # Start camera update in a background thread
            # daemon=True means thread will exit when main thread exits
            Thread(target=self.update_camera, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")

    def stop_camera(self):
        """
        Stop the camera feed and clean up resources.
        
        Releases the camera, resets UI elements, and resets all detection
        result labels to their default states.
        """
        self.camera_active = False  # Signal update_camera to stop
        
        # Release the camera resource
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Reset button to show "Live Camera" and green color
        self.camera_btn.config(text="Live Camera", bg='#2ecc71')
        
        # Reset all UI labels to default state
        self.image_label.config(image='', text="No image loaded")
        self.result_label.config(text="Predicted: -", fg='black')
        self.confidence_label.config(text="Confidence: -", fg='black')
        self.status_label.config(text="Status: -", fg='black')

    def update_camera(self):
        """
        Continuously capture frames from camera and perform real-time detection.
        
        This runs in a background thread and:
        1. Captures frames from the camera
        2. Classifies each frame
        3. Detects object region
        4. Draws bounding boxes
        5. Updates GUI with results
        
        Runs until self.camera_active is set to False.
        """
       
        while self.camera_active:
            # Capture a frame from the camera
            ret, frame = self.camera.read()
            if not ret:
                break  # Exit if frame capture failed
            
            # Store current frame for potential screenshot
            self.current_frame = frame.copy()
            
            # ====== Process Frame ======
            # Convert from BGR (OpenCV) to RGB (PIL) color space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image
            pil = Image.fromarray(rgb)
            
            # Classify the image using the model
            cls, conf = self.classify_image(pil)
            
            # Detect the object bounding box
            bbox = self.detect_object_region(pil)
            
            # Draw detection results on image
            img_box, status = self.draw_detection_box(pil, cls, conf, bbox)
            
            # Update result labels in GUI
            self.update_results(cls, conf, status)
            
            # ====== Display Image in GUI ======
            # Resize image to fit display area while maintaining aspect ratio
            img_box.thumbnail((700, 380), Image.Resampling.LANCZOS)
            
            # Convert PIL Image to PhotoImage for Tkinter display
            photo = ImageTk.PhotoImage(img_box)
            
            # Update image label with new frame
            self.image_label.config(image=photo, text='')
            self.image_label.image = photo  # Keep reference to prevent garbage collection

    def upload_image(self):
        """
        Open file dialog to select an image and classify it.
        
        Steps:
        1. Stop camera if running
        2. Show file browser to select image
        3. Classify the selected image
        4. Display results with bounding box
        """
        # Stop camera if it's currently active
        self.stop_camera()
        
        # Open file dialog to select image
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        # If user cancelled the dialog, exit
        if not path:
            return
        
        try:
            # Open image and convert to RGB
            img = Image.open(path).convert('RGB')
            
            # Classify the image
            cls, conf = self.classify_image(img)
            
            # Detect object region in image
            bbox = self.detect_object_region(img)
            
            # Draw bounding box and label on image
            img_box, status = self.draw_detection_box(img.copy(), cls, conf, bbox)
            
            # Display the annotated image
            self.display_image(img_box)
            
            # Update result labels
            self.update_results(cls, conf, status)
            
            # Force GUI update
            self.root.update()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def capture_and_classify(self):
        """
        Capture the current camera frame and classify it.
        
        This is useful for pausing camera feed on a specific frame
        and getting detailed analysis of that frame.
        """
        
        if self.current_frame is None:
            return
        
        # Convert camera frame from BGR to RGB
        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        
        # Stop camera feed
        self.stop_camera()
        
        # Classify the captured frame
        cls, conf = self.classify_image(img)
        
        # Detect object region
        bbox = self.detect_object_region(img)
        
        # Draw detection results
        img_box, status = self.draw_detection_box(img, cls, conf, bbox)
        
        # Display and update results
        self.display_image(img_box)
        self.update_results(cls, conf, status)

    def save_screenshot(self):
        """
        Save the current frame from camera with detection results as a JPEG file.
        
        Saves to disk with filename: detection_screenshot_N.jpg
        where N is an incrementing counter.
        """
        
        if self.current_frame is None:
            return
        
        # Increment screenshot counter for unique filename
        self.screenshot_count += 1
        fname = f"detection_screenshot_{self.screenshot_count}.jpg"
        
        # Convert camera frame from BGR to RGB
        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        
        # Classify and detect objects
        cls, conf = self.classify_image(pil)
        bbox = self.detect_object_region(pil)
        
        # Draw detection results
        pil_box, _ = self.draw_detection_box(pil, cls, conf, bbox)
        
        # Save the annotated image
        pil_box.save(fname)
        
        # Show confirmation message
        messagebox.showinfo("Screenshot Saved", f"Saved as {fname}")

    def display_image(self, image):
        """
        Display an image in the GUI image label.
        
        Resizes the image to fit the display area while maintaining aspect ratio
        and converts it to PhotoImage format for Tkinter display.
        
        Args:
            image (PIL.Image): Image to display
        """
        # Create a copy to avoid modifying the original
        img = image.copy()
        
        # Resize to fit display area (max 700x380 pixels)
        img.thumbnail((700, 380), Image.Resampling.LANCZOS)
        
        # Convert PIL Image to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(img)
        
        # Update label with new image
        self.image_label.config(image=photo, text='')
        self.image_label.image = photo  # Keep reference to prevent garbage collection

    def update_results(self, cls, conf, status):
        """
        Update the results display labels with classification results.
        
        Colors the output based on recognition status:
        - Green for recognized objects (confidence >= threshold)
        - Red for unknown objects (confidence < threshold)
        
        Args:
            cls (str): Predicted class name
            conf (float): Confidence score (0-100)
            status (str): Status string ("DETECTED" or "UNKNOWN")
        """
        
        # Check if object is recognized
        is_rec = conf >= self.CONFIDENCE_THRESHOLD * 100
        
        # Format predicted class text
        if is_rec:
            predicted_text = f"Predicted: {cls.replace('_', ' ')}"
        else:
            predicted_text = "Predicted: Unknown"
        
        # Format confidence text
        confidence_text = f"Confidence: {conf:.1f}%"
        
        # Format status text with symbols
        if is_rec:
            status_text = "Status: ✓ DETECTED"
        else:
            status_text = "Status: ✗ UNKNOWN"
        
        # Set color based on recognition status
        text_color = '#27ae60' if is_rec else '#e74c3c'  # Green or Red
        
        # Update all labels with new values
        self.result_label.config(text=predicted_text, fg=text_color)
        self.confidence_label.config(text=confidence_text, fg=text_color)
        self.status_label.config(text=status_text, fg=text_color)
        
        # Force GUI update immediately
        self.root.update_idletasks()

    def on_closing(self):
        """
        Handle application closing event.
        
        Ensures camera is properly released and window is destroyed
        when user closes the application.
        """
        # Clean up camera resources
        self.stop_camera()
        
        # Close the main window
        self.root.destroy()


# Application Entry Point
if __name__ == "__main__":
    # Create root Tkinter window
    root = tk.Tk()
    
    # Initialize the application
    app = OfficeObjectClassifier(root)
    
    # Set the window close handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI event loop
    root.mainloop()