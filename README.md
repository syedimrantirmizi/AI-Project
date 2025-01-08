# Vehicle Type Classification

This project is a machine learning-based application for classifying images of vehicles into four categories: **Bus**, **Car**, **Truck**, and **Motorcycle**. The solution uses a Convolutional Neural Network (CNN) with Transfer Learning (MobileNetV2) to ensure high accuracy and scalability.

## Features

- **Model Training:** Utilizes Transfer Learning with MobileNetV2 to classify vehicle images. Handles class imbalance with computed class weights and includes data augmentation (rotation, flipping, zooming).
- **Prediction:** Predicts labels for single images and visualizes predictions with true labels and confidence scores.
- **Random Image Testing:** Selects random images from the dataset and evaluates predictions with visual outputs.
- **Data Preparation:** Renames files within folders for easier organization and processing.
- **Evaluation:** Computes a confusion matrix and classification report to analyze model performance.

## Dataset

The dataset used for this project can be downloaded from the following link:
[Download Dataset](<https://drive.google.com/drive/folders/1hpdSMkNkjEXOOlL6qusAZnkRv6Rf9RED>)

## Folder Structure

- `dataset`: Contains subfolders for each vehicle type (`Bus`, `Car`, `Truck`, `Motorcycle`).
- `randomimagestest`: Stores output images for random image testing.
- `singleimagetest`: Stores output images for single image predictions.
- **Scripts:**
  - `model.py`: Model training and saving.
  - `randomImages.py`: Random image selection, prediction, and output generation.
  - `rename.py`: Script to rename dataset images.
  - `testing.py`: Single image prediction and visualization.
  - `ConfusionMatrix.py`: Evaluates the model using a confusion matrix and classification report.

## Requirements

### Libraries

- Python 3.7-3.10
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- PIL (Pillow)

### Installation

```bash
git clone https://github.com/syedimrantirmizi/AI-Project
cd VehicleTypeClassification
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow
```

## How to Use

### 1. Train the Model

```bash
python model.py
```

This script:

- Loads the dataset.
- Applies data augmentation and class balancing.
- Trains a CNN using MobileNetV2.
- Saves the trained model as `vehicle_classifier.h5`.

### 2. Predict Single Image

```bash
python testing.py
```

Modify `test_image_path` to specify your image. Generates visual output in `singleimagetest`.

### 3. Test Random Images 

```bash
python randomImages.py
```

Selects 10 random images and generates:

- `output.png`: Displays all selected images with true and predicted labels.
- `confidence_scores.png`: Lists confidence scores for each image.

### 4. Rename Dataset Images (Skip if you are using my dataset)

```bash
python rename.py
```

Renames all images in the `Truck` folder.

### 5. Evaluate the Model

```bash
python ConfusionMatrix.py
```

Computes a confusion matrix and displays a classification report.

## Outputs

- **Training Plots:** Accuracy and validation accuracy over epochs.
- **Predicted Images:**
  - `randomimagestest/output.png`: True vs Predicted labels for multiple images.
  - `randomimagestest/confidence_scores.png`: Confidence scores for each image.
- **Confusion Matrix:** A heatmap of the confusion matrix for performance evaluation.

## Example Outputs

### Predicted Labels

```plaintext
True Label: Car
Predicted Label: Truck
Confidence Scores:
Bus: 2.34%
Car: 89.45%
Truck: 5.67%
Motorcycle: 2.54%
```

### Confusion Matrix

![Confusion Matrix Heatmap](confusionMatrix.png)

## Authors

This project was developed by:

- **Syed Imran Tirmizi**
- **Subhan Akhter**
- **Hashir Naveed**
- **Izhan Rehan**
