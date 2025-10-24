#  Census Income Classification using PyTorch

This project builds a **binary classification model** to predict whether an individual earns more than **$50,000** annually using the **Census Income Dataset**.  
It demonstrates an end-to-end **machine learning pipeline** — including data preprocessing, encoding, model training, evaluation, and real-time prediction.

##  Repository Structure

```
├── census_income_classification.ipynb # Jupyter Notebook with full workflow
├── model.pth # Saved PyTorch model
├── encoder.joblib # Encoders for categorical features
├── scaler.joblib # Scaler for numerical features
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── data/ # Dataset folder (if applicable)

```

##  Project Overview

The goal of this project is to **classify income level** (`>50K` or `<=50K`) based on demographic and employment data.  
This classification helps in socio-economic analysis, targeted policy-making, and automated eligibility screening systems.

##  Features

- Preprocessing and cleaning of categorical and numerical columns<br>
- One-hot encoding for categorical variables<br>
- Label encoding for target variable<br>
- Feature scaling using StandardScaler<br>
- Fully connected neural network built using **PyTorch**<br>
- Model evaluation using accuracy, precision, recall, and F1-score<br>
- Real-time prediction for custom user input<br>

---

##  Model Architecture

The model is a **feedforward neural network (ANN)** with:<br>
- Input Layer: 108 neurons (after encoding)<br>
- Hidden Layers: Two layers with ReLU activation<br>
- Output Layer: Single neuron with Sigmoid activation for binary classification<br>

**Optimizer:** Adam<br>
**Loss Function:** Binary Cross-Entropy Loss (BCELoss)<br>


## ⚙️ Installation

Clone the repository and install dependencies:

```
git clone https://github.com/yourusername/census-income-classification.git
cd census-income-classification
pip install -r requirements.txt
```

**Dataset Information**<br>
The dataset used is the Adult Census Income Dataset from the UCI Machine Learning Repository.

Target Column:
income: <=50K or >50K

Key Features:
age, workclass, education, marital-status, occupation, relationship, race, sex, hours-per-week, native-country, etc.

**Model Training**
Run the notebook or script to train the model:

python train_model.py
Training steps include:

1.Encoding categorical columns using LabelEncoder and OneHotEncoder
2.Splitting dataset into train/test sets
3.Standardizing numerical features
4.Building and training the PyTorch neural network
5.Evaluating performance metrics
6.Saving the trained model and encoders

**Model Evaluation**
Example results after training:

**Metric	Value**
Accuracy	86.4%
Precision	84.7%
Recall	82.1%
F1 Score	83.3%

**Making Predictions**
Once the model is trained and saved, you can make predictions for new individuals:

```
import torch
import joblib
import pandas as pd
from model import IncomeClassifier  # Import model class

# Load saved model and encoders
model = IncomeClassifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()

encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Example input data
new_data = {
    'age': 37,
    'workclass': 'Private',
    'education': 'Bachelors',
    'marital.status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'hours.per.week': 60,
    'native.country': 'United-States'
}

# Convert to DataFrame
df = pd.DataFrame([new_data])

# Encode and scale input, then predict
# (Ensure preprocessing matches training pipeline)
prediction = model(torch.tensor(scaled_input).float())
print("Predicted Income:", ">50K" if prediction.item() > 0.5 else "<=50K")

```
**Saving & Loading the Model**
After training:
```
torch.save(model.state_dict(), 'model.pth')
```
**To load later:**
```
model.load_state_dict(torch.load('model.pth'))
model.eval()
```
**Requirements**
All dependencies are listed in requirements.txt:

```
torch
pandas
numpy
scikit-learn
joblib
matplotlib
```
**Install them using:**
```
pip install -r requirements.txt
```
