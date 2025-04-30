# Pneumonia Detection Web App

A deep learning-powered web application that classifies chest X-rays as **Pneumonia**, **Normal**, or **Uncertain**, and visualizes the model's attention using **Grad-CAM** heatmaps.

## Project Overview

This application leverages a fine-tuned **ResNet-18** convolutional neural network to detect pneumonia from chest X-ray images. It features:

- A **Flask** backend for model inference
- A user-friendly **HTML frontend** for image upload and results display
- **Batch management** to store and review previous predictions
- **Grad-CAM visualization** to highlight important regions in X-ray images for interpretability

## Model Details

- **Architecture:** ResNet-18 with modified fully connected layers
- **Training:** Conducted on the UCSD/Guangzhou chest X-ray dataset
- **Input:** Resized to 224×224, normalized using ImageNet stats
- **Output:** Binary classification (Normal vs Pneumonia) with confidence score

## Dataset

- Sourced from [https://data.mendeley.com/datasets/rscbjbr9sj/3)
- Contains: 5,212 training, 20 validation, 624 test images
- Preprocessed: Resized, grayscale to RGB, augmented during training

## Features

- Upload one or multiple X-rays for analysis
- Get predictions and confidence percentages
- View **Grad-CAM** heatmaps to interpret model decisions
- Save predictions into named batches
- Navigate and delete stored test results

## Usage
### 1. Remember to change the home directory where the folder for test, training, and validation will be stored.

### 2. Add Model File

Ensure the file `model.pth` is in the root directory. This contains the trained weights of the ResNet-18 model. This is achieved by running the backend:
```bash
python New.ipynb.py
```
This saves the optimal model after 10 epochs
### 3. Run the App
run the front end:
```bash
python pneumonia_api.py
```

Visit `http://localhost:5000` in your browser.

### 4. UI Controls

- Upload X-rays via the main page
- Store results in batches
- View batches via `/view_batches`
- Shut down the app from the UI or POST to `/quit`

## Project Structure

```
├── New.ipynb                # used for model training (backend)
├── pneumonia_api.py         # Flask app and model inference logic
├── model.pth                # Trained ResNet-18 model weights
├── static/
│   ├── uploads/             # Uploaded original X-ray images
│   ├── heatmaps/            # Grad-CAM heatmaps
│   └── batches/             # Stored result folders
├── templates/
│   ├── index.html           # Main page
│   ├── view_batches.html    # Batch overview
│   └── view_tests.html      # View stored test predictions
└── results.json             # JSON file for tracking predictions
```

## Evaluation

- **Balanced Accuracy**: 0.87
- **Normal Accuracy**: 175/234
- **Pneumonia Accuracy**: 368/22
- Slight bias due to class imbalance
- Model interpretable with Grad-CAM overlays

## Future Work

- Incorporate deeper architectures like **DenseNet**
- Add multi-class classification (e.g., COVID-19)
- Host app via cloud services for public access
