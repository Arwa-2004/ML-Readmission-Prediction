# Pateint readmission prediction using logistic regression

This project was created to test my skills in machine learning and data analysis using a self created small data. It predicts whether a patient will be readmitted to the hospital based on features like age, BMI, blood pressure, and medical history. It uses logistic regression which is used for binary classification, and the model's performance is evaluated using accuracy and a classification report.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Results](#results)
4. [Future Work](#future-work)
5. [License](#license)

## Introduction
Hospital readmission is a critical issue in healthcare. This project aims to predict readmission using machine learning. The model is trained on a sample dataset and evaluated using accuracy and a classification report.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ML-Readmission-Prediction.git


## Results
   Accuracy: 1.00
   
Classification Report:

                precision     recall   f1-score    support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1
    accuracy                           1.00         2
    macro avg      1.00     1.00       1.00         2
    weighted avg   1.00     1.00       1.00         2
    
Accuracy is 100% due to the small data. Precision, Recall, and F1-score are  100% for both classes (0 and 1)
Precision: Every time the model predicted a class, it was correct. Recall: The model identified all actual instances correctly. F1-score: The balance of precision and recall is perfect.


## Future Work 
Using a real-world dataset.
Experiment with other machine learning models like Random Forest.

   
