# Insertion of the required Python libraries

from st_pages import show_pages_from_config, add_page_title
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

# Modelling (Random Forest Classifier)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Modelling (SVC Classifier)

from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Tree Visualisation

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from IPython.display import display

# Either this or add_indentation() MUST be called on each page in your

add_page_title()

# Construct the full path to data.csv
csv_file_path = '/home/piphs/Software-Engineering/src/pages/data.csv'

# Load the data from CSV into a DataFrame
try:
    data = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print("Error: File 'data.csv' not found.")

# ------------------------------------------- RANDOM FOREST CLASSIFIER ---------------------------------------------------

# Pre-processing the data for the Random Forest Classifier
features = ['smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'diagnosis']
cancer_breast_selection = data[features]

# Splitting the data into features and target
cancer_breast_selection.loc[:, 'diagnosis'] = cancer_breast_selection['diagnosis'].map({'M': 1, 'B': 0})

# Replace NaN values with the mean of the respective columns
cancer_breast_selection.fillna(cancer_breast_selection.mean(), inplace=True)

# Replace the values in other selected columns:
for col in ['smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean']:
    cancer_breast_selection[col] = cancer_breast_selection[col].map({'M': 1, 'B': 0, 'unknown': 0})

# Split the data into features and target
X = cancer_breast_selection.drop('diagnosis', axis=1)
y = cancer_breast_selection['diagnosis']

# Drop rows with NaN values in y
X = X.dropna()
y = y.dropna()

# Check if there are enough samples to split into training and testing sets
if len(X) == 0 or len(y) == 0:
    print("Error: Not enough samples to split into training and testing sets.")
else:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ------------------------------------------- EVALUATION FOR THE MODEL (RF) ----------------------------------------------

    # First instance of the RF model with default parameters
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Test the accuracy of the model
    y_pred = rf.predict(X_test)

    # Print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Visualize the first 4 trees of the Random Forest Classifier
    for i in range(4):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(tree,feature_names=X_train.columns, filled=True, max_depth=2, impurity=False, proportion=True)
        graph = graphviz.Source(dot_data)
        display(graph)
