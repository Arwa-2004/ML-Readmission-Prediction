import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset 
data = {
    'Age': [45, 60, 30, 50, 70, 25, 80, 55, 40, 65],
    'BMI': [22.5, 27.8, 24.0, 30.2, 28.5, 23.1, 31.0, 26.5, 25.0, 29.3],
    'Blood_Pressure': [120, 140, 110, 135, 150, 115, 155, 130, 125, 145],
    'Diabetes': [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    'Heart_Disease': [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    'Previous_Admissions': [2, 5, 0, 3, 7, 1, 8, 4, 2, 6],
    'Readmitted': [1, 1, 0, 1, 1, 0, 1, 0, 0, 1]  # Target variable
}

# Convert to df
df = pd.DataFrame(data)
print(df)

# Split into features X and target y
X = df.drop('Readmitted', axis=1)
print(X)
y = df['Readmitted']

# Split into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))
