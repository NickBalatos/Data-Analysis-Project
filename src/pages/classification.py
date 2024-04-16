# Insertion of the required Python libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from st_pages import show_pages_from_config, add_page_title
import os
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


# Modelling (Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import plot_tree
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

# Modelling (SVC Classifier)
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Tree VisualisationInitial Information
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from IPython.display import display

# Either this or add_indentation() MUST be called on each page in your
add_page_title()

# ------------------------------HEADER----------------------------------

st.write("""
# Random Forest (Τυχαίο Δάσος):

Το Random Forest είναι ένα ισχυρό μοντέλο μηχανικής μάθησης που χρησιμοποιείται 
για προβλέψεις και ταξινόμηση. Αποτελείται από πολλά δέντρα αποφάσεων που λειτουργούν 
ανεξάρτητα μεταξύ τους και συνδυάζουν τις προβλέψεις τους για να δώσουν το τελικό αποτέλεσμα. 
Η βασική ιδέα πίσω από το Random Forest είναι η δύναμη του συνόλου: η συλλογή των διαφορετικών 
δέντρων αποφάσεων επιτρέπει την αντιμετώπιση του overfitting και την αύξηση της ακρίβειας των προβλέψεων.
""")

# ------------------------------HEADER----------------------------------


# ------------------------------CSV-------------------------------------

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
    exit()

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
    exit()

# ------------------------------CSV------------------------------------

# Plotting the SVC Results
def plot_svc_results(y_test, y_pred_svc):
    plt.rc('font', size=20)
    plt.figure(figsize=(20, 10))
    cm = confusion_matrix(y_test, y_pred_svc)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix - SVC")
    st.pyplot(plt)

# --------------------------MAIN FUNCTION------------------------------

def main():
    # Set font size for matplotlibimport seaborn as sns
    plt.rc('font', size=300)

    global X, y  # declare X, y as global variables to access them within main()

    # Pre-process the data
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2)

# ------------------RANDOM FOREST CLASSIFIER---------------------------

    # Add a button to run the Random Forest algorithm
    if st.button("Run Random Forest"):

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
            plt.figure(figsize=(30, 20))
            plot_tree(rf.estimators_[i], feature_names=X.columns, filled=True, max_depth=2, impurity=False, proportion=True)
            st.pyplot(plt)
        
# ------------------RANDOM FOREST CLASSIFIER--------------------------



# ------------------------------HEADER--------------------------------

    # Initial Information about SVC
    st.write("""
    # Support Vector Classifier SVC (Ταξινομητής Διανυσματικής Υποστήριξης):
    Ο Support Vector Classifier είναι ένας δυνατός αλγόριθμος ταξινόμησης που 
    χρησιμοποιείται για δυαδική ταξινόμηση. Ο αλγόριθμος επιχειρεί να διαχωρίσει
    τα δεδομένα στον χώρο των χαρακτηριστικών με τον καλύτερο δυνατό τρόπο, δημιουργώντας
    ένα υπερεπίπεδο που ορίζει μια διαίρεση μεταξύ των κλάσεων. Ο SVC είναι αρκετά ευέλικτος
    και αποτελεσματικός σε πολλά προβλήματα ταξινόμησης, ενώ είναι επίσης ανθεκτικός σε δεδομένα μεγάλης διάστασης.""")

# ------------------------------HEADER--------------------------------
    

# ------------------SUPPORT VECTOR CLASSIFIER SVC---------------------


    # Add a button to run the SVC algorithm
    if st.button("Run SVC"):

        # Instantiate the SVC model
        svc = SVC()

        # Train the model
        svc.fit(X_train, y_train)

        # Test the accuracy of the model
        y_pred_svc = svc.predict(X_test)

        # Print the accuracy of the model
        accuracy_svc = accuracy_score(y_test, y_pred_svc)
        st.write("SVC Accuracy:", accuracy_svc)

        # Plot the confusion matrix
        plot_svc_results(y_test, y_pred_svc)

# ------------------SUPPORT VECTOR CLASSIFIER SVC---------------------


if __name__ == "__main__":
    main()

# --------------------------MAIN FUNCTION----------------------------