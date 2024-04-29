

Alzheimer's Disease Prediction using Image Processing
This project aims to predict Alzheimer's disease using image processing techniques on MRI images. The dataset consists of MRI images segmented into categories: Very mild demented, non-demented, mild demented, and moderate demented. Various machine learning models, including CNN 2D, Random Forest, MLP, and SVM, were applied to the dataset to predict Alzheimer's disease.

Table of Contents
Dataset
Code Files
Model Performance
Requirements
Usage
Results
License
Dataset
The dataset used in this project contains MRI images segmented into four categories: Very mild demented, non-demented, mild demented, and moderate demented. The images were preprocessed and normalized before being used for training and testing the models.

Code Files
preprocess.py: Contains functions for preprocessing the MRI images.
models.py: Defines the CNN, Random Forest, MLP, and SVM models used for prediction.
train.py: Trains the models on the preprocessed dataset.
evaluate.py: Evaluates the performance of the trained models using various metrics.
Model Performance
CNN 2D: Accuracy - 54%
Random Forest: Accuracy - 85%
MLP: Accuracy - 70%
SVM: Accuracy - 62%
Combined Model (RF and MLP): Accuracy - 87%
Stacking (RF, MLP, and SVM): Accuracy - 94%
Requirements
Python 3.x
TensorFlow
scikit-learn
NumPy
pandas
matplotlib
Usage
Clone the repository: git clone https://github.com/your-username/alzheimers-prediction.git
Install the required dependencies: pip install -r requirements.txt
Preprocess the dataset: python preprocess.py
Train the models: python train.py
Evaluate the models: python evaluate.py
Results
The project achieved a maximum accuracy of 94% using feature extraction and a stacking approach with Random Forest, MLP, and SVM models. The trained models can be used to predict Alzheimer's disease from MRI images with a high degree of accuracy.

License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
