import streamlit as st
from st_pages import show_pages_from_config, add_page_title
import numpy as np# for testing reasons
import pandas as pd
from sklearn.cluster import ( 
      KMeans,
      AgglomerativeClustering
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (
      calinski_harabasz_score,
      davies_bouldin_score,
      silhouette_score,
)

add_page_title()


def get_data():
      """
      Load the dataset that was uploaded in the homepage
      """
      try:
      # Check if data is already loaded in session state
            if 'data' in st.session_state:
                  if any(st.session_state.data.dtypes.apply(lambda x: pd.api.types.is_string_dtype(x))):
                    raise TypeError("Dataset contains string.")
                  return st.session_state.data
            else:
                  raise KeyError("Data is not loaded in session state.")            
      except TypeError:
            st.error("Το αρχείο δεδομένων περιέχει γράμματα. Φορτώστε ένα αρχείο που περιέχει μόνο αριθμητικές τιμές.")
            exit()
      except KeyError:
            st.error("Δεν υπάρχουν δεδομένα. Φορτώστε αρχείο CSV ή Excel στο Home Tab.")
            exit()



def k_means(clusters: int, data: pd.DataFrame) -> tuple[bool, np.ndarray]:
      """
      Perform K Means clustering on the given data.

      Parameters:
            clusters (int): The number of clusters to create.
            data (pandas.DataFrame): The input data points.

      Returns:
            tuple: A tuple containing:
                  - bool: True, indicating successful execution of the algorithm.
                  - labels (numpy.ndarray): The labels assigned to each data point by the clustering algorithm.
      """
      kmeans = KMeans(n_clusters= clusters)
      labels = kmeans.fit_predict(data)

      return True, labels
      
      
def hierarchical_clustering(clusters: int, data: pd.DataFrame) -> tuple[bool, np.ndarray]:
      """
      Perform hierarchical clustering on the given data.

      Parameters:
            clusters (int): The number of clusters to create.
            data (pandas.DataFrame): The input data points.

      Returns:
            tuple: A tuple containing:
                  - bool: True, indicating successful execution of the algorithm.
                  - labels (numpy.ndarray): The labels assigned to each data point by the clustering algorithm.
      Notes:
            The function uses the Agglomerative Clustering type, with Euclidean distance and Ward's method as linkage.
      """
      hierarchical_cluster = AgglomerativeClustering(n_clusters=clusters, metric='euclidean', linkage='ward')
      labels = hierarchical_cluster.fit_predict(data)

      return True, labels


def calc_dunn_index(data: pd.DataFrame, labels: np.ndarray) -> float:
      """
      Calculate the Dunn index for a given clustering.

      Parameters:
            data (pandas.DataFrame): The input data points.
            labels (numpy.ndarray): The labels assigned to each data point by the clustering algorithm.

      Returns:
            float: The calculated Dunn index value.
      """

      distances = euclidean_distances(data)
      W = np.max([np.max(distances[labels == label][:, labels == label]) for label in np.unique(labels)])
      centroids = [np.mean(data[labels == label], axis=0) for label in np.unique(labels)]
      B = np.min([np.min([np.linalg.norm(centroids[i] - centroids[j]) for j in range(len(centroids)) if j != i]) for i in range(len(centroids))])
      Dunn_index = B / W
      return Dunn_index


def calc_bss(data: pd.DataFrame, labels: np.ndarray) -> float:
      """
      Calculate the Between-Cluster Sum of Squares (BSS) for a given clustering.

      Parameters:
            data (pandas.DataFrame): The input data points.
            labels (numpy.ndarray): The labels assigned to each data point by the clustering algorithm.

      Returns:
            float: The calculated Between-Cluster Sum of Squares (BSS) value.
      """

      centroids = [np.mean(data[labels == label], axis=0) for label in np.unique(labels)]
      overall_centroid = np.mean(data, axis=0)
      BSS = sum([np.sum((centroid - overall_centroid) ** 2) * np.sum(labels == label) for centroid, label in zip(centroids, np.unique(labels))])
      return BSS

      
def calculate_metrics(data: pd.DataFrame, labels: np.ndarray):
      """
      Calculate clustering evaluation metrics.

      Parameters:
            data (pandas.DataFrame): The input data points.
            labels (numpy.ndarray): The labels assigned to each data point by the clustering algorithm.
      """
      st.write(f"Silhouette Score: {silhouette_score(data, labels)}")
      st.write(f"Calinski-Harabasz Index: {calinski_harabasz_score(data, labels)}")
      st.write(f"Davies-Bouldin Index: {davies_bouldin_score(data, labels)}")
      st.write(f"Dunn Index: {calc_dunn_index(data, labels)}")
      #wss
      st.write(f"BSS: {calc_bss(data, labels)}")
      # nmi ; For NMI we need ground truth labels. 'Ground truth' is that data or information that you have that is 'true' or assumed to be true.
      # That means that you have high or perfect knowledge of what it is.
      # ari ; The same applies to ARI.
      # homogeneity, completeness, v_measure ; The same applies again to homogeneity, completeness, v_measure
      # cluster purity ; The same again applies to Cluster Purity



def main():
      data = get_data()
      
      # Get all the features columns except the class
      features = list(data.columns)[:-1]

      # Get the features data
      data = data[features]


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
      kmeans_clusters = st.number_input(label="Αριθμός ομάδων", min_value=1, key= "num_kmeans")
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
      hierar_clusters = st.number_input(label="Αριθμός ομάδων", min_value=1, key= "num_hier")
      st.button("Run", key="hier_clust")
            

      #-------------------RESULTS AND COMPRARISON--------------------------
      st.title("Results and Comprarison")
      # We use a session state variables because we want the values to be maintained across the session
      if st.session_state.kmeans:
            st.session_state.flag_kmeans, st.session_state.labels_kmeans = k_means(kmeans_clusters, data)
      if st.session_state.hier_clust:
            st.session_state.flag_hier_clust, st.session_state.labels_hier_clust = hierarchical_clustering(hierar_clusters, data)


      # If the algorithm has been executed, we set a flag to true.
      # This flag is used to ensure that the metrics are displayed throughout the session.
      st.header("K-Means Results")
      if st.session_state.flag_kmeans:
            calculate_metrics(data, st.session_state.labels_kmeans)
      st.header("Hierarchical Clustering (Agglomerative Clustering)")
      if st.session_state.flag_hier_clust:
            calculate_metrics(data, st.session_state.labels_hier_clust)




if __name__ == "__main__":
      main()




