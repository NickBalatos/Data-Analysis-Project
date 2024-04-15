# Insertion of the required Python libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from st_pages import show_pages_from_config, add_page_title
import os

# Modelling (Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import plot_tree

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


def main():
    # Set font size for matplotlib
    plt.rc('font', size=300)

    # Construct the full path to data.csv
    csv_file_path = '/home/piphs/Software-Engineering/src/pages/wdbc_processed.csv'

    # Load the data from CSV into a DataFrame
    try:
        data = pd.read_csv(csv_file_path, index_col=0, names=["ID", "Diagnosis", "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", 
                                                "Smoothness Mean", "Compactness Mean", "Concavity Mean", "Concave Points Mean",
                                                "Symmetry Mean", "Fractal Dimension Mean", "Radius SE", "Texture SE", "Perimeter SE",
                                                "Area SE", "Smoothness SE", "Compactness SE", "Concavity SE", "Concave Points SE",
                                                "Symmetry SE", "Fractal Dimension SE", "Radius Worst", "Texture Worst", "Perimeter Worst",
                                                "Area Worst", "Smoothness Worst", "Compactness Worst", "Concavity Worst", "Concave Points Worst",
                                                "Symmetry Worst", "Fractal Dimension Worst", "Label"])
    except FileNotFoundError:
        st.error("Error: File 'wdbc_processed.csv' not found.")
        return

    data.dropna(subset=['Label'], inplace=True)

    # Convert the Diagnosis column to binary
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

    # Pre-processing the data
    X = data.drop('Label', axis=1)
    y = data['Label']

    # Drop rows with NaN values
    y = y.dropna()

    # Check if there are enough samples to split into training and testing sets
    if len(X) == 0 or len(y) == 0:
        st.error("Error: Not enough samples to split into training and testing sets.")
        return

    # Add a button to run the Random Forest algorithm
    if st.button("Run Random Forest"):

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Instantiate the Random Forest model
        rf = RandomForestClassifier()

        # Train the model
        rf.fit(X_train, y_train)

        # Test the accuracy of the model
        y_pred = rf.predict(X_test)

        # Print the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Visualize the first 3 trees of the Random Forest Classifier
        for i in range(3):
            plt.figure(figsize=(15, 8))
            plot_tree(rf.estimators_[i], feature_names=X_train.columns, filled=True, max_depth=2, impurity=False, proportion=True)
            st.pyplot(plt)

if __name__ == "__main__":
    main()
