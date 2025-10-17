import torch
from torchvision import transforms, models
import torch.nn.functional as F # Import F for softmax
from PIL import Image
import cv2
import numpy as np
import os

# --- Configuration (Must match your Jupyter setup) ---
INPUT_SIZE = 224 
NUM_CLASSES = 10 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_model_weights.pth' 

# This list MUST match the order ImageFolder assigned during training!
CLASS_NAMES = ['Book', 'Bottle', 'Calculator', 'File_Folder', 'Keyboard', 'Mouse', 'Mug', 'Pen_Pencil', 'Stapler', 'Tape_Dispenser']
print(f"Loading classes in this order: {CLASS_NAMES}")

# --- 1. Define Image Transformations ---
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(INPUT_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. Load Trained Model ---
def load_model():
    # Load ResNet-50 model structure
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze layers (or unfreeze layer4, depending on final training setup)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES) 
    
    # NOTE: You may need to manually unfreeze layer4 here if your final training used fine-tuning
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    
    # Load the trained weights
    if os.path.exists(MODEL_PATH):
        # NOTE: Using weights_only=False for compatibility with older saves
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH} and set to {device}")
        return model
    else:
        print(f"CRITICAL ERROR: Model weights not found at {MODEL_PATH}. Cannot run live classification.")
        return None

# --- 3. Main Live Stream Function (with Bounding Box) ---
def live_classification(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Live Classification Running (Press 'q' to quit) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Convert OpenCV BGR frame to RGB for PyTorch processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess frame and move to device
        image_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            # Use F.softmax imported from torch.nn.functional
            probabilities = F.softmax(outputs, dim=1)[0] 
            confidence, predicted_index = torch.max(probabilities, 0)
            
            predicted_class = CLASS_NAMES[predicted_index.item()]
            confidence_percent = confidence.item() * 100
        
        # --- VISUAL ENHANCEMENTS (Bounding Box and Text) ---
        
        # 1. Define the center box (50% of the frame) for the area of focus
        h, w, _ = frame.shape
        box_w, box_h = int(w * 0.5), int(h * 0.5)

        # Calculate top-left (x1, y1) and bottom-right (x2, y2) corners
        x1 = int((w - box_w) / 2)
        y1 = int((h - box_h) / 2)
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # 2. Draw the fixed green rectangle (B, G, R)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 3. Format and place the classification text (just above the box)
        label = f"{predicted_class}: {confidence_percent:.2f}%"
        # Text position: (x1, y1 - 10) is just above the top-left corner of the box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('Office Item Live Classifier', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# --- Execution ---
if __name__ == '__main__':
    loaded_model = load_model()
    if loaded_model:
        live_classification(loaded_model)