# Brain CT Hemorrhage and Fracture Classification

## Description
This project implements a *Convolutional Neural Network (CNN) for multi-label classification of brain CT images to detect various types of hemorrhages (Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural) and fractures, as well as identifying cases with no hemorrhage. The dataset includes CT images and a CSV file (`hemorrhage_diagnosis.csv`) with corresponding labels. The model is built using TensorFlow/Keras, trained with early stopping, and evaluated using accuracy and a classification report. Training history is visualized with accuracy and loss plots.

## Features
- Dataset: Custom brain CT dataset with images organized by patient and slice number, labeled via CSV.
- Model: CNN with convolutional layers, batch normalization, max pooling, and dense layers for multi-label classification.
- Labels: Predicts 7 binary labels (5 hemorrhage types, no hemorrhage, fracture).
- Training: Uses Adam optimizer, binary cross-entropy loss, and early stopping to prevent overfitting.
- Evaluation: Reports test accuracy and detailed classification metrics (precision, recall, F1-score).
- Visualization*: Plots training/validation accuracy and loss curves.

## Prerequisites
- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `opencv-python` (`cv2`)
  - `tensorflow`
  - `scikit-learn`
  - `matplotlib`
- Google Colab with Google Drive access (or local setup with dataset)
- Brain CT dataset (images and `hemorrhage_diagnosis.csv`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-ct-classification.git
   cd brain-ct-classification

##Install dependencies:
pip install numpy pandas opencv-python tensorflow scikit-learn matplotlib

##For Colab:
Upload the script to Google Colab.
Mount Google Drive (see code: drive.mount("/content/drive")).

##Dataset:
Place the dataset in Google Drive (e.g., /content/drive/MyDrive/Patients_CT/) or locally.

##Expected structure:

Patients_CT/
├── train/
│   ├── 1/brain/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── 2/brain/
│   └── ...
├── validation/
├── test/
├── hemorrhage_diagnosis.csv

##Usage
#Prepare the Dataset:
  Ensure the dataset is in the correct folder structure (see above).
  The CSV (hemorrhage_diagnosis.csv) should have columns: PatientNumber, SliceNumber, Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural, No_Hemorrhage,             
  Fracture_Yes_No.

#Run the Script:
#In Colab:
  Upload script.py to Colab.
  Update data_folder path if needed (e.g., /content/drive/MyDrive/Patients_CT).
  Run all cells.
#Locally:
python script.py

#This will:
Load and preprocess images (resize to 224x224, normalize).
Train the CNN model for up to 30 epochs with early stopping.
Evaluate on the test set and print accuracy/classification report.
Display training/validation accuracy and loss plots.

##Code Structure
Data Loading: Reads images from train, validation, and test folders, matches with CSV labels.
Preprocessing: Resizes images to 224x224, normalizes pixel values to [0, 1].
Model:
  CNN with 3 conv layers (16, 32, 64 filters), batch normalization, and max pooling.
  Dense layers with dropout (0.6) and sigmoid activation for 7 outputs.
Training: Uses binary cross-entropy loss, Adam optimizer, and early stopping (patience=5).
Evaluation: Computes test accuracy and classification report (precision, recall, F1-score).
Visualization: Plots accuracy/loss curves using Matplotlib.

##Results
Test Accuracy: Printed after evaluation (e.g., Test accuracy: 0.85).
Classification Report: Shows precision, recall, and F1-score for each label.
Plots: Displays training/validation accuracy and loss over epochs.

##Example
For a test image, the model might predict:
Labels: [0, 1, 0, 0, 0, 0, 1] (Intraparenchymal hemorrhage and fracture detected).

##Notes
Dataset Access: The dataset is assumed to be private. Update paths if using a different dataset.
Colab vs. Local: The code includes Colab-specific
Drive mounting. For local runs, adjust data_folder.
Hyperparameters: Batch size (8), learning rate (0.001), and dropout (0.6) can be tuned for better performance.
Image Quality: Ensure images are readable; the script skips corrupted files.

#Contributing
Fork the repository.
Create a feature branch (git checkout -b feature-branch).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.
