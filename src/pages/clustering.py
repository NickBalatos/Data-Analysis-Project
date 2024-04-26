import streamlit as st
from st_pages import show_pages_from_config, add_page_title
import numpy as np# for testing reasons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import (
      calinski_harabasz_score,
      davies_bouldin_score,
      silhouette_score,
      normalized_mutual_info_score,
      adjusted_rand_score,
      homogeneity_completeness_v_measure
)

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

def get_data(x, y, data):
      # It should fetch the csv data from the home page
      # because the home page isn't ready, it will generate the required data
      local_x = np.random.uniform(1, 10, 100) # rows from the csv file
      local_y = np.random.uniform(1, 10, 100) # the target 

      local_x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
      local_y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

      
      # Append each item to the originally passed lists
      x.extend(local_x)
      y.extend(local_y)
      data.extend(list(zip(x, y)))


def interface():
      x = []
      y = []
      data = []
      get_data(x, y, data)

      
      dataset = pd.read_csv("data.csv")
      # Get all the features columns except the class
      features = list(dataset.columns)[:-2]

      # Get the features data
      data = dataset[features]
      # Class
      y = list(dataset.columns)[len(dataset.columns)-1]


      # Initialize session state variables if not already present
      if 'flag_kmeans' not in st.session_state:
            st.session_state.flag_kmeans = False
      if 'flag_hier_clust' not in st.session_state:
            st.session_state.flag_hier_clust = False
      if "labels_kmeans" not in st.session_state:
            st.session_state.labels_kmeans = None
      if "labels_hier_clust" not in st.session_state:
            st.session_state.labels_hier_clust = None


      #----------------------------K MEANS--------------------------------
      st.title("K-Means")
      st.write(
            """
            Ο αλγόριθμος K-Means είναι ένας clustering αλγόριθμος μηχανικής μάθησης που χρησιμοποιείται στη στατιστική ανάλυση για την ομαδοποίηση δεδομένων σε έναν προκαθορισμένο αριθμό ομάδων, k.
            Ο αλγόριθμος επιλέγει τυχαία k κέντρα και στη συνέχεια αναθέτει κάθε σημείο δεδομένων στην πλησιέστερη ομάδα, βελτιστοποιώντας τη θέση των κέντρων μέχρι να σταθεροποιηθούν.
            Χρησιμοποιείται ευρέως για την ανίχνευση μοτίβων, την ανάλυση ομάδων και ως προπαρασκευαστικό βήμα για άλλες αλγοριθμικές εφαρμογές.
            """
      )
      kmeans_clusters = st.number_input(label="Αριθμός ομάδων", min_value=1, max_value=5, key= "num_kmeans")
      st.button("Run", key="kmeans")
            

      #--------------------HIERARCHIAL CLUSTERING-------------------------
      st.title("Hierarchical Clustering (Agglomerative Clustering)")
      st.write(
            """
            Ο αλγόριθμος Hierarchical Clustering είναι μια μέθοδος ομαδοποίησης δεδομένων που χτίζει ιεραρχικά συστήματα ομάδων.
            Ο Hierarchical Clustering ξεκινάει θεωρώντας κάθε σημείο δεδομένων ως μια ξεχωριστή ομάδα και στη συνέχεια, επαναληπτικά ενώνει τις πιο κοντινές ομάδες μέχρι να επιτευχθεί μια μόνο ομάδα ή ο στόχος αριθμός των ομάδων.
            Υπάρχουν δύο κύριοι τύποι: Agglomerative (συγκεντρωτικό), που ξεκινά με μικρές ομάδες και τις συνδυάζει, και Divisive (διαιρετικό), που ξεκινά με μία ολική ομάδα και τη διαιρεί.
            """
      )
      hierar_clusters = st.number_input(label="Αριθμός ομάδων", min_value=1, max_value=5, key= "num_hier")
      st.button("Run", key="hier_clust")
            

      #-------------------RESULTS AND COMPRARISON--------------------------
      st.title("Results and Comprarison")
      # We use a session state variables because we want the values to be maintained across the session
      if st.session_state.kmeans:
            st.session_state.flag_kmeans, st.session_state.labels_kmeans = k_means(kmeans_clusters, data)
      if st.session_state.hier_clust:
            st.session_state.flag_hier_clust, st.session_state.labels_hier_clust = hierarchical_clustering(hierar_clusters, data)


      # If the algorithm has been executed, we set a flag to true.
      # This flag is used to ensure that graphs are displayed throughout the session.
      st.header("K-Means Results")
      if st.session_state.flag_kmeans:
            calculate_metrics(data, st.session_state.labels_kmeans)
      st.header("Hierarchical Clustering (Agglomerative Clustering)")
      if st.session_state.flag_hier_clust:
            calculate_metrics(data, st.session_state.labels_hier_clust)




def k_means(clusters, data):
      # Run K Means
      kmeans = KMeans(n_clusters= clusters)
      labels = kmeans.fit_predict(data)

      return True, labels # Successful execution of the algorithm
      
      

def hierarchical_clustering(clusters, data):
      # Run Hierarchial Clustering
      hierarchical_cluster = AgglomerativeClustering(n_clusters=clusters, metric='euclidean', linkage='ward')
      labels = hierarchical_cluster.fit_predict(data)

      return True, labels # Successful execution of the algorithm

      
def calculate_metrics(data, labels):
      silhouette_avg = silhouette_score(data, labels)
      calinski_score = calinski_harabasz_score(data, labels)
      davies_score = davies_bouldin_score(data, labels)
      # Dunn Index
      # WSS
      # BSS
      #nmi = normalized_mutual_info_score(true_labels, predicted_labels)
      #ari = adjusted_rand_score(true_labels, predicted_labels)
      #homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, predicted_labels)
      # Cluster Purity
      st.write(f"Silhouette Score: {silhouette_avg}")
      st.write(f"Calinski-Harabasz Index: {calinski_score}")
      st.write(f"Davies-Bouldin Index: {davies_score}")



# General main function of the file should call all the necessary function
interface()




