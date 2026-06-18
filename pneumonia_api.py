from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import cv2
import numpy as np
import torch.nn as nn
import time
import pydicom
from torchvision.models import resnet18
import torch.nn.functional as F
import json
import sqlite3
from contextlib import closing
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Ensure static storage for uploaded images
UPLOAD_FOLDER = "static/uploads"
BATCHES_FOLDER = "static/batches"
HEATMAP_FOLDER = "static/heatmaps"
for folder in (HEATMAP_FOLDER, UPLOAD_FOLDER, BATCHES_FOLDER):
    os.makedirs(folder, exist_ok=True)

# Predictions are persisted in SQLite. This is process-safe across gunicorn workers,
# unlike the previous shared in-memory dict + full JSON rewrite on every request.
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_db()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                   filename   TEXT PRIMARY KEY,
                   prediction TEXT NOT NULL,
                   created_at TEXT NOT NULL
               )"""
        )


def save_prediction(filename, prediction):
    with closing(_db()) as conn, conn:
        conn.execute(
            "INSERT INTO predictions (filename, prediction, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(filename) DO UPDATE SET "
            "prediction = excluded.prediction, created_at = excluded.created_at",
            (filename, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def get_prediction(filename):
    with closing(_db()) as conn:
        row = conn.execute(
            "SELECT prediction FROM predictions WHERE filename = ?", (filename,)
        ).fetchone()
    return row["prediction"] if row else None


init_db()
# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}

# Trained ResNet-18 (pneumonia head). Loaded lazily and cached on first use so the module
# can be imported (e.g. by tests) without the weights file present or a load on every import.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
_model = None


def build_model():
    m = resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 2),  # Raw logits; softmax is applied once in predict()
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    m.eval()
    return m


def get_model():
    """Lazily load and cache the trained model on first use."""
    global _model
    if _model is None:
        _model = build_model()
    return _model

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load DICOM images
def load_dicom(file):
    dicom = pydicom.dcmread(file)
    image = dicom.pixel_array  # Extract pixel data
    return Image.fromarray(image)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        # Hook to get gradients                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self):
        grad = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (grad * self.activation).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap).squeeze().detach().cpu().numpy()

        heatmap = cv2.resize(heatmap, (224, 224))  # Resize to image dimensions
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
        return heatmap

    def overlay_heatmap(self, image, heatmap, alpha=0.5, output_size=(150, 150)):
        image = np.array(image)  # Convert PIL image to NumPy array

    # Ensure image is 3-channel BGR (convert RGB to BGR for OpenCV)
        if len(image.shape) == 2:  # If grayscale, convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:  # If RGB, convert to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Normalize heatmap and convert it to 3-channel
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, output_size)
        image = cv2.resize(image, output_size)

    # Blend images
        superimposed_img = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)

        return superimposed_img

    
    
    
    
# Pneumonia prediction function
def predict(image: Image.Image, filename: str):
    model = get_model()
    image_tensor = transform(image).unsqueeze(0)  # Convert image to tensor
    
    # Initialize Grad-CAM
    target_layer = model.layer4[-1]  # Last convolutional layer
    grad_cam = GradCAM(model, target_layer)

    # Forward pass
    image_tensor.requires_grad = True
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1).squeeze()
    prediction = torch.argmax(probabilities).item()
    confidence = probabilities[prediction].item() * 100

    if confidence < 70:
        return "Uncertain Image", "invalid", confidence, None

    label = "PNEUMONIA" if prediction == 1 else "NORMAL"

    # Backward pass for Grad-CAM
    model.zero_grad()
    class_score = output[0, prediction]
    class_score.backward()

    # Generate and overlay heatmap
    heatmap = grad_cam.generate_heatmap()
    heatmap_img = grad_cam.overlay_heatmap(image, heatmap)

  

# Generate a unique filename
    heatmap_filename = f"{os.path.splitext(filename)[0]}_gradcam.jpg"
    heatmap_path = os.path.join("static/heatmaps", heatmap_filename)
    cv2.imwrite(heatmap_path, heatmap_img)

    return label, "valid", confidence, heatmap_path


@app.route('/')
def upload_form():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict_pneumonia():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    files = request.files.getlist('file')  # Allow multiple uploads
    if not files or all(file.filename == "" for file in files):
        return render_template('index.html', error="No selected files.")

    predictions = []
    stored_files = []
    grad_cam_images = []

    for file in files:
        if not allowed_file(file.filename):
            predictions.append((file.filename, "Invalid file type"))
            continue

        # Sanitize the user-supplied filename to prevent path traversal.
        filename = secure_filename(file.filename)
        if not filename:
            predictions.append((file.filename, "Invalid file name"))
            continue

        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except UnidentifiedImageError:
            predictions.append((filename, "Invalid image file"))
            continue

        save_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(save_path)
        stored_files.append(filename)

        result, _, confidence, heatmap_path = predict(image, filename)
        prediction_text = f"{result} ({confidence:.2f}% confidence)"
        predictions.append((filename, prediction_text))

        # Persist the prediction (SQLite upsert; safe across workers)
        save_prediction(filename, prediction_text)

        if heatmap_path:
            grad_cam_images.append((file.filename, heatmap_path))

    return render_template(
        'index.html',
        predictions=predictions, 
        grad_cam_images=grad_cam_images, 
        stored_files=stored_files
    )




@app.route('/store_tests', methods=['POST'])
def store_tests():
    folder_name = secure_filename(request.args.get('folder', 'default_batch')) or 'default_batch'
    batch_folder = os.path.join(BATCHES_FOLDER, folder_name)

    if os.path.exists(batch_folder):
        return jsonify({"success": False, "message": "A batch with this name already exists. Please enter a different name."}), 400
    elif not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    batch_results = []  # List to store image details
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for image in os.listdir(UPLOAD_FOLDER):
        source = os.path.join(UPLOAD_FOLDER, image)
        destination = os.path.join(batch_folder, image)
        os.rename(source, destination)  # Move image to batch folder

        # Copy stored predictions for this image
        grad_cam_filename = f"{os.path.splitext(image)[0]}_gradcam.jpg"
        grad_cam_source = os.path.join("static/heatmaps", grad_cam_filename)
        grad_cam_destination = os.path.join(batch_folder, grad_cam_filename)

        if os.path.exists(grad_cam_source):  # Check if Grad-CAM image exists before moving
            os.rename(grad_cam_source, grad_cam_destination)
        
        # Retrieve the stored prediction for this image
        prediction_text = get_prediction(image) or "No prediction found in records."

        batch_results.append({
            "filename": image,
            "prediction": prediction_text,
            "grad_cam": grad_cam_filename if os.path.exists(grad_cam_destination) else None,
            "timestamp": timestamp
        })

    # Save batch-specific predictions in results.json inside the batch folder
    batch_results_path = os.path.join(batch_folder, "results.json")
    with open(batch_results_path, "w") as f:
        json.dump(batch_results, f, indent=4)

    return jsonify({"success": True, "message": f"Batch '{folder_name}' stored successfully on {timestamp}!"})

@app.route('/view_batches')
def view_batches():
    batches = os.listdir(BATCHES_FOLDER)
    return render_template('view_batches.html', batches=batches)

@app.route('/delete_batch/<batch_name>', methods=['POST'])
def delete_batch(batch_name):
    batch_name = secure_filename(batch_name)
    batch_folder = os.path.join(BATCHES_FOLDER, batch_name)
    if batch_name and os.path.exists(batch_folder):
        for file in os.listdir(batch_folder):
            os.remove(os.path.join(batch_folder, file))
        os.rmdir(batch_folder)
    return jsonify({"success": True})

@app.route('/view_tests/<batch_name>')
def view_tests(batch_name):
    batch_name = secure_filename(batch_name)
    batch_folder = os.path.join(BATCHES_FOLDER, batch_name)
    images = os.listdir(batch_folder) if os.path.exists(batch_folder) else []

    # Load predictions from the batch's results.json
    batch_results_path = os.path.join(batch_folder, "results.json")
    if os.path.exists(batch_results_path):
        with open(batch_results_path, "r") as f:
            batch_results = json.load(f)
    else:
        batch_results = []

    return render_template('view_tests.html', images=batch_results, batch_name=batch_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
