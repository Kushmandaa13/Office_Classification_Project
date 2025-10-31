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


# CLASS: OfficeObjectClassifier (Main Program)
class OfficeObjectClassifier:
    def __init__(self, root):
        
        #Initialize the main application window and variables.
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
        self.INPUT_SIZE = 224
        self.CONFIDENCE_THRESHOLD = 0.70  # 70% confidence threshold

        # Load trained model
        self.model = self.load_model()

        # Test-Time Augmentation (TTA) Transforms
        self.tta_transforms = [
            #Original Image
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),

            #Horizontally Flipped
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),

            #Slight Rotation
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        ]
        
        # Camera Variables
        self.camera = None
        self.camera_active = False
        self.current_frame = None
        self.screenshot_count = 0

        # Build the Tkinter UI
        self.create_widgets()

    # MODEL LOADING FUNCTION
    def load_model(self):
        
        try:
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))
            model.load_state_dict(torch.load('best_model_weights.pth', map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default pretrained weights for demo...")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))
            model = model.to(self.device)
            model.eval()
            return model

    #UI LAYOUT CREATION

    def create_widgets(self):
        
        # Title section
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        tk.Label(title_frame, text="Office Object Classifier & Detector",
                 font=('Helvetica', 24, 'bold'), bg='#2c3e50', fg='white').pack(expand=True)
        tk.Label(title_frame, text="ResNet-50 | Real-time Detection | Office Objects",
                 font=('Helvetica', 10), bg='#2c3e50', fg='#ecf0f1').pack()

        # Main content
        content_frame = tk.Frame(self.root, bg='#f0f4f8')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Buttons for image and camera
        button_frame = tk.Frame(content_frame, bg='#f0f4f8')
        button_frame.pack(pady=10)
        self.upload_btn = tk.Button(button_frame, text="Upload Image",
                                    font=('Helvetica', 14, 'bold'),
                                    bg='#3498db', fg='white',
                                    width=15, height=2, cursor='hand2',
                                    command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.camera_btn = tk.Button(button_frame, text="Live Camera",
                                    font=('Helvetica', 14, 'bold'),
                                    bg='#2ecc71', fg='white',
                                    width=15, height=2, cursor='hand2',
                                    command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=1, padx=10)

        # Display area
        self.display_frame = tk.Frame(content_frame, bg='white', relief='solid', borderwidth=2, height=400)
        self.display_frame.pack(pady=20, fill='both', expand=False)
        self.image_label = tk.Label(self.display_frame, text="No image loaded",
                                    font=('Helvetica', 14), bg='white', fg='gray')
        self.image_label.pack(expand=True)

        tk.Label(content_frame, text="🟢 Green = Recognized | 🔴 Red = Unknown (<70%)",
                 font=('Helvetica', 9), bg='#f0f4f8', fg='#555').pack(pady=5)

        # Camera controls
        self.camera_controls_frame = tk.Frame(content_frame, bg='#f0f4f8')
        tk.Button(self.camera_controls_frame, text="📸 Capture & Detect",
                  font=('Helvetica', 12, 'bold'), bg='#e74c3c', fg='white',
                  width=18, height=2, cursor='hand2',
                  command=self.capture_and_classify).pack(side='left', padx=5)
        tk.Button(self.camera_controls_frame, text="💾 Save Screenshot",
                  font=('Helvetica', 12, 'bold'), bg='#f39c12', fg='white',
                  width=18, height=2, cursor='hand2',
                  command=self.save_screenshot).pack(side='left', padx=5)
        self.fps_label = tk.Label(content_frame, text="FPS: 0.0",
                                  font=('Helvetica', 10), bg='#f0f4f8', fg='#27ae60')

        # Results display
        self.result_frame = tk.Frame(content_frame, bg='#ecf0f1', relief='solid', borderwidth=2)
        self.result_frame.pack(fill='x', pady=10)
        tk.Label(self.result_frame, text="Detection Result",
                 font=('Helvetica', 12, 'bold'), bg='#ecf0f1').pack(pady=5)
        self.result_label = tk.Label(self.result_frame, text="Predicted Class: -",
                                     font=('Helvetica', 14), bg='#ecf0f1')
        self.result_label.pack(pady=5)
        self.confidence_label = tk.Label(self.result_frame, text="Confidence: -",
                                         font=('Helvetica', 12), bg='#ecf0f1')
        self.confidence_label.pack(pady=5)
        self.status_label = tk.Label(self.result_frame, text="Status: -",
                                     font=('Helvetica', 11, 'bold'), bg='#ecf0f1')
        self.status_label.pack(pady=5)

    #IMAGE DETECTION & CLASSIFICATION
    def detect_object_region(self, image):

        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            h_img, w_img = img_array.shape[:2]
            area = w * h
            img_area = h_img * w_img
            if 0.05 < (area / img_area) < 0.95:
                pad = 20
                x, y = max(0, x - pad), max(0, y - pad)
                w, h = min(w_img - x, w + 2 * pad), min(h_img - y, h + 2 * pad)
                return (x, y, x + w, y + h)

        # Default bounding box (if contour fails)
        h, w = img_array.shape[:2]
        m = int(min(h, w) * 0.1)
        return (m, m, w - m, h - m)

    def draw_detection_box(self, image, class_name, confidence, bbox):
        
        draw = ImageDraw.Draw(image)
        is_recognized = confidence >= self.CONFIDENCE_THRESHOLD * 100
        color = (0, 255, 0) if is_recognized else (255, 0, 0)
        label = f"{class_name.replace('_', ' ')}: {confidence:.1f}%" if is_recognized else f"Unknown: {confidence:.1f}%"
        status = "DETECTED" if is_recognized else "UNKNOWN"
        draw.rectangle(bbox, outline=color, width=4)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        tb = draw.textbbox((0, 0), label, font=font)
        draw.rectangle([bbox[0], bbox[1] - tb[3] - 10, bbox[0] + tb[2] + 10, bbox[1]], fill=color)
        draw.text((bbox[0] + 5, bbox[1] - tb[3] - 5), label, fill=(255, 255, 255), font=font)
        return image, status

    def classify_image(self, image):
        
        try:
            self.model.eval()
            all_probs = []

            with torch.no_grad():
                for transform in self.tta_transforms:
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    outputs = self.model(image_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    all_probs.append(probs)

            avg_probs = torch.mean(torch.cat(all_probs, dim=0), dim=0)
            confidence, predicted_index = torch.max(avg_probs, 0)
            predicted_class = self.class_names[predicted_index.item()]
            confidence_percent = confidence.item() * 100

            return predicted_class, confidence_percent
        except Exception as e:
            print(f"Classification error: {e}")
            return "Error", 0.0

    #CAMERA FUNCTIONS
    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):

        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return
            self.camera_active = True
            self.camera_btn.config(text="Stop Camera", bg='#e74c3c')
            self.camera_controls_frame.pack(pady=10)
            self.fps_label.pack(pady=5)
            Thread(target=self.update_camera, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")

    def stop_camera(self):
    
        self.camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_btn.config(text="Live Camera", bg='#2ecc71')
        self.camera_controls_frame.pack_forget()
        self.fps_label.pack_forget()
        self.image_label.config(image='', text="No image loaded")
        self.result_label.config(text="Predicted Class: -", fg='black')
        self.confidence_label.config(text="Confidence: -", fg='black')
        self.status_label.config(text="Status: -", fg='black')

    def update_camera(self):
       
        while self.camera_active:
            ret, frame = self.camera.read()
            if not ret:
                break
            self.current_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            cls, conf = self.classify_image(pil)
            bbox = self.detect_object_region(pil)
            img_box, status = self.draw_detection_box(pil, cls, conf, bbox)
            self.update_results(cls, conf, status)
            img_box.thumbnail((700, 380), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_box)
            self.image_label.config(image=photo, text='')
            self.image_label.image = photo


    #IMAGE HANDLING
    def upload_image(self):
        self.stop_camera()
        path = filedialog.askopenfilename(title="Select an image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            img = Image.open(path).convert('RGB')
            cls, conf = self.classify_image(img)
            bbox = self.detect_object_region(img)
            img_box, status = self.draw_detection_box(img.copy(), cls, conf, bbox)
            self.display_image(img_box)
            self.update_results(cls, conf, status)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def capture_and_classify(self):
        
        if self.current_frame is None:
            return
        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.stop_camera()
        cls, conf = self.classify_image(img)
        bbox = self.detect_object_region(img)
        img_box, status = self.draw_detection_box(img, cls, conf, bbox)
        self.display_image(img_box)
        self.update_results(cls, conf, status)

    def save_screenshot(self):
        
        if self.current_frame is None:
            return
        self.screenshot_count += 1
        fname = f"detection_screenshot_{self.screenshot_count}.jpg"
        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cls, conf = self.classify_image(pil)
        bbox = self.detect_object_region(pil)
        pil_box, _ = self.draw_detection_box(pil, cls, conf, bbox)
        pil_box.save(fname)
        messagebox.showinfo("Screenshot Saved", f"Saved as {fname}")

    def display_image(self, image):
        img = image.copy()
        img.thumbnail((700, 380), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo, text='')
        self.image_label.image = photo


    def update_results(self, cls, conf, status):
        
        is_rec = conf >= self.CONFIDENCE_THRESHOLD * 100
        self.result_label.config(text=f"Predicted: {cls if is_rec else 'Unknown'}",
                                 fg='#27ae60' if is_rec else '#e74c3c')
        self.confidence_label.config(text=f"Confidence: {conf:.2f}%",
                                     fg='#27ae60' if is_rec else '#e74c3c')
        self.status_label.config(text=f"Status: {'✓' if is_rec else '✗'} {status}",
                                 fg='#27ae60' if is_rec else '#e74c3c')

    #EXIT FUNCTION
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


#MAIN ENTRY POINT
if __name__ == "__main__":
    root = tk.Tk()
    app = OfficeObjectClassifier(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()  