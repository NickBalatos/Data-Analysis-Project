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

def get_data():
      # Load the dataset that was uploaded in the homepage
      try:
      # Check if data is already loaded in session state
            if 'data' in st.session_state:
                  return st.session_state.data
            else:
                  raise KeyError("Data is not loaded in session state.")
      except KeyError as e:
            st.error(f"Error: {e}")
            exit()

# ------------------------------CSV-------------------------------------
    


# ---------------------PLOTTING THE SVC RESULTS--------------------------

# Plotting the SVC Results
def plot_svc_results(X_test, y_test, y_pred_svc):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_svc)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix - SVC")
    st.pyplot(plt)

    # Plot the scatter plot for SVC predictions
    plot_svc_scatter(X_test, y_test, y_pred_svc)
    
# Plot the scatter plot for SVC predictions
def plot_svc_scatter(X_test, y_test, y_pred_svc):
    plt.figure(figsize=(10, 8))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=100, marker='x', linewidths=3, edgecolors='k', label='True labels')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_svc, cmap=plt.cm.coolwarm, s=50, alpha=0.5, label='Predicted labels')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter Plot - SVC Predictions")
    plt.legend()
    st.pyplot(plt)

# ---------------------PLOTTING THE SVC RESULTS--------------------------


# --------------------------MAIN FUNCTION------------------------------

def main():

    # Load the data
    data = get_data()
    
    # Get all the features columns except the class
    features = list(data.columns)[:-1]

    # Get the features data
    X = data[features]
    y = data['label']  # Assuming 'Label' column contains target labels

    # Drop rows with NaN values
    X = X.dropna()

    # Concatenate the features and labels into a single DataFrame
    data_with_labels = pd.concat([X, y], axis=1)

    # Check if there are enough samples to split into training and testing sets
    if len(X) == 0 or len(y) == 0:
        st.error("Error: Not enough samples to split into training and testing sets.")
        exit()

    # Set font size for matplotlibimport seaborn as sns
    plt.rc('font', size=10)

    # Pre-processing the data
    imputer_X = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer_X.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2)


# ------------------RANDOM FOREST CLASSIFIER---------------------------

    # Add a slider for selecting the number of trees to print
    num_trees = st.slider("Επιλέξτε τον αριθμό των δέντρων που θέλετε να εκτυπωθούν:", 1, 100, 3)

    # Add a button to run the Random Forest algorithm
    if st.button("Run Random Forest"):

        # Instantiate the Random Forest model
        rf = RandomForestClassifier()

        # Train the model
        rf.fit(X_train, y_train)

        # Test the accuracy of the model
        y_pred_rf = rf.predict(X_test)

        # Print the accuracy of the model
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write("Accuracy:", accuracy_rf)

        # Visualize the selected number of trees from the Random Forest Classifier
        for i in range(min(num_trees, len(rf.estimators_))):
            plt.figure(figsize=(30, 20))
            plot_tree(rf.estimators_[i], feature_names=data.columns, filled=True, max_depth=2, impurity=False, proportion=True)
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

    # Add a slider for selecting the value of C
    C = st.slider("Επιλέξτε την τιμή που θα έχει το C:", 0.1, 10.0, step=0.1)

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
        plot_svc_results(X_test, y_test, y_pred_svc)



# ------------------SUPPORT VECTOR CLASSIFIER SVC---------------------


if __name__ == "__main__":
    main()

# --------------------------MAIN FUNCTION----------------------------
