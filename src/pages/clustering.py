import streamlit as st
from st_pages import show_pages_from_config, add_page_title
import numpy as np# for testing reasons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA


import asyncio


import csv



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

      fig_kmeans = None

      x = 0
      y = 0
      #----------------------------K MEANS--------------------------------
      st.title("K-Means")
      st.write(
            """
            Ο αλγόριθμος K-Means είναι ένας clustering αλγόριθμος μηχανικής μάθησης που χρησιμοποιείται στη στατιστική ανάλυση για την ομαδοποίηση δεδομένων σε έναν προκαθορισμένο αριθμό ομάδων, k.
            Ο αλγόριθμος επιλέγει τυχαία k κέντρα και στη συνέχεια αναθέτει κάθε σημείο δεδομένων στην πλησιέστερη ομάδα, βελτιστοποιώντας τη θέση των κέντρων μέχρι να σταθεροποιηθούν.
            Χρησιμοποιείται ευρέως για την ανίχνευση μοτίβων, την ανάλυση ομάδων και ως προπαρασκευαστικό βήμα για άλλες αλγοριθμικές εφαρμογές.
            """
      )
      clusters = st.number_input(label="Αριθμός ομάδων", min_value=1, max_value=5, key= "num_kmeans")
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
      clusters = st.number_input(label="Αριθμός ομάδων", min_value=1, max_value=5, key= "num_hier")
      if st.button("Run", key="hier_clust"):
            hierarchical_clustering(clusters, data, x, y)   
            fig_hierar = None
            
      #-------------------RESULTS AND COMPRARISON--------------------------
      st.title("Results and Comprarison")
      if "kmeans" in st.session_state and st.session_state.kmeans:
            k_means(clusters, data, x, y)
      



def k_means(clusters, data, x, y):
      # Run K Means
      kmeans = KMeans(n_clusters= clusters)
      labels = kmeans.fit_predict(data)
      # Appling dimensional reduction in order to plot clusters from multi-feature dataset
      pca = PCA(2)
      data_2d = pca.fit_transform(data)

      plotted_data = data_2d
      plotted_labels = labels      
      # Plotting
      fig, ax = plt.subplots()
      sc = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
      st.pyplot(fig)

def hierarchical_clustering(clusters, data, x, y):
      # Run Hierarchial Clustering
      linkage_data = linkage(data, method='ward', metric='euclidean')
      hierarchical_cluster = AgglomerativeClustering(n_clusters=clusters, metric='euclidean', linkage='ward')
      labels = hierarchical_cluster.fit_predict(data)
      # Appling dimensional reduction in order to plot clusters from multi-feature dataset
      pca = PCA(2)
      data_2d = pca.fit_transform(data)
      # Plot dendogram
      fig, ax = plt.subplots()
      ax.set_title("Dendogram")
      dendrogram(linkage_data)
      st.pyplot(fig)
      # Plot the clusters in a scatter plot
      fig2, ax2 = plt.subplots()
      ax2.set_title("Scatterplot")
      ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
      st.pyplot(fig2)



# General main function of the file should call all the necessary function
interface()




