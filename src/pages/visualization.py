from st_pages import show_pages_from_config, add_page_title
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

def main():
    
   # Checking for data in session state-------------
    if 'data' not in st.session_state:
        st.error("Δεν υπάρχουν δεδομένα. Φορτώστε αρχείο CSV ή Excel στο Home Tab.")
        return
    
    data = st.session_state.data
    st.write("Τα φορτωμένα δεδομένα:")
    st.write(data)
    
    st.markdown("""---""")

    # Executing PCA-------------
    st.header("Αλγόριθμοι Μείωσης Διάστασης") 
    st.header("PCA")
    st.write("Ο αλγόριθμος μείωσης διάστασης PCA (Principal Component Analysis) είναι ένας από τους πιο δημοφιλείς και ευρέως χρησιμοποιούμενους αλγορίθμους στον χώρο της μηχανικής μάθησης και της ανάλυσης δεδομένων. Ο στόχος του PCA είναι η μείωση της διάστασης των δεδομένων, διατηρώντας ταυτόχρονα τη μεγαλύτερη δυνατή ποσότητα πληροφορίας. Ο PCA λειτουργεί με την εκτίμηση των κύριων συνιστωσών (principal components) των δεδομένων, οι οποίες είναι γραμμικοί συνδυασμοί των αρχικών χαρακτηριστικών. Οι κύριες συνιστώσες ταξινομούνται με βάση τη διακύμανση των δεδομένων, με την πρώτη κύρια συνιστώσα να περιέχει τη μεγαλύτερη διακύμανση, η δεύτερη τη δεύτερη μεγαλύτερη, και ούτω καθεξής.")
    st.write("Οπτικοποίηση δεδομένων με τον PCA:")

    pca_reduced_data = PCA(n_components=2).fit_transform(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_reduced_data[:, 0], pca_reduced_data[:, 1], alpha=0.6)
    plt.title('2D Visualization using PCA')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    st.pyplot(plt)
    
    # Converting non-numeric data to NaN-------------
    numeric_data = data.select_dtypes(include=[np.number])
    # Droppng NaN collumns
    numeric_data = numeric_data.dropna()

    # Executing t-SNE-------------
    st.header("t-SNE")
    st.write("Ο αλγόριθμος μείωσης διάστασης t-SNE (t-distributed Stochastic Neighbor Embedding) είναι ένας αλγόριθμος που χρησιμοποιείται για την οπτικοποίηση και την εξερεύνηση πολυδιάστατων δεδομένων σε έναν χαμηλότερης διάστασης χώρο. Η βασική ιδέα πίσω από το t-SNE είναι η μετατροπή υψηλής διάστασης δεδομένων σε χαμηλής διάστασης αναπαραστάσεις, διατηρώντας τις αποστάσεις μεταξύ των δεδομένων όσο το δυνατόν πιο κοντά στις αρχικές. Συνήθως χρησιμοποιείται σε συνδυασμό με άλλες τεχνικές οπτικοποίησης ή ανάλυσης δεδομένων για την κατανόηση και την ανάδειξη συσχετίσεων μεταξύ των παρατηρούμενων φαινομένων.")
    st.write("Οπτικοποίηση δεδομένων με τον t-SNE:")

    tsne_reduced_data = TSNE(n_components=2).fit_transform(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_reduced_data[:, 0], tsne_reduced_data[:, 1], alpha=0.6)
    plt.title('2D Visualization using t-SNE')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    st.pyplot(plt)

    st.markdown("""---""")

    # EDA Diagrams-------------
    st.header("Διαγράμματα EDA")
    st.write("Το EDA (Exploratory Data Analysis) διάγραμμα είναι ένα γραφικό που χρησιμοποιείται για την εξερεύνηση και την ανάλυση δεδομένων. Συνήθως χρησιμοποιείται στην αρχική φάση ενός έργου δεδομένων για να αναδείξει μοτίβα, τάσεις και ανωμαλίες. Τα EDA διαγράμματα μπορεί να περιλαμβάνουν ιστόγραμματα, διαγράμματα διασποράς, γραφήματα κουτιών και άλλα, που βοηθούν στην καλύτερη κατανόηση των δεδομένων και στη λήψη αποφάσεων.")

    # Box Plot Diagram
    st.subheader("Διάγραμμα Κουτιών (Box Plot)")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    st.pyplot(plt)

    # Density Diagram
    st.subheader("Διάγραμμα Πυκνότητας (Density Diagram)")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, shade=True)
    st.pyplot(plt)

    # Heatmap
    st.subheader("Διάγραμμα Κατακερματισμού (Heatmap)")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

    
if __name__ == "__main__":
    main()
