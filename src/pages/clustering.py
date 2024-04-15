import streamlit as st
from st_pages import show_pages_from_config, add_page_title
import numpy as np# for testing reasons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from kneed import KneeLocator
import random
import string

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


      st.title("K-Means")
      st.write(
            """
            Ο αλγόριθμος K-Means είναι ένας clustering αλγόριθμος μηχανικής μάθησης που χρησιμοποιείται στη στατιστική ανάλυση για την ομαδοποίηση δεδομένων σε έναν προκαθορισμένο αριθμό ομάδων, k.
            Ο αλγόριθμος επιλέγει τυχαία k κέντρα και στη συνέχεια αναθέτει κάθε σημείο δεδομένων στην πλησιέστερη ομάδα, βελτιστοποιώντας τη θέση των κέντρων μέχρι να σταθεροποιηθούν.
            Χρησιμοποιείται ευρέως για την ανίχνευση μοτίβων, την ανάλυση ομάδων και ως προπαρασκευαστικό βήμα για άλλες αλγοριθμικές εφαρμογές.
            """
      )
      st.write(recommended_clusters(data))
      clusters = st.number_input(label="Αριθμός ομάδων", min_value=1, max_value=5)
      if st.button("Run", key="kmeans"):
            st.write("Run K-Means")
            k_means(clusters, data, x, y)


      st.title("Hierarchical Clustering (Agglomerative Clustering)")
      st.write(
            """
            Ο αλγόριθμος Hierarchical Clustering είναι μια μέθοδος ομαδοποίησης δεδομένων που χτίζει ιεραρχικά συστήματα ομάδων.
            Ο Hierarchical Clustering ξεκινάει θεωρώντας κάθε σημείο δεδομένων ως μια ξεχωριστή ομάδα και στη συνέχεια, επαναληπτικά ενώνει τις πιο κοντινές ομάδες μέχρι να επιτευχθεί μια μόνο ομάδα ή ο στόχος αριθμός των ομάδων.
            Υπάρχουν δύο κύριοι τύποι: Agglomerative (συγκεντρωτικό), που ξεκινά με μικρές ομάδες και τις συνδυάζει, και Divisive (διαιρετικό), που ξεκινά με μία ολική ομάδα και τη διαιρεί.
            """
      )
      clusters = st.number_input(label="Algorithm parameter", min_value=1, max_value=5)
      if st.button("Run", key="hier_clust"):
            hierarchical_clustering(clusters, data, x, y)



def recommended_clusters(data):
      K_range = range(1, 11)
      sse = []
      for i in K_range:
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)
      knee_locator = KneeLocator(K_range, sse, curve = "convex", direction = "decreasing")
      optimal_k = knee_locator.knee
      return f'Βέλτιστος αριθμός ομάδων(clusters): {optimal_k}'


def k_means(clusters, data, x, y):
      kmeans = KMeans(n_clusters= clusters)
      labels = kmeans.fit_predict(data)
      # plt.scatter(x, y, c=labels)
      fig, ax = plt.subplots()
      sc = ax.scatter(x, y, c=labels)
      ax.set_title("Scatterplot")
      st.pyplot(fig) 
      # arr = np.random.normal(1, 1, size=100)
      # fig, ax = plt.subplots()
      # ax.hist(arr, bins=20)

      # st.pyplot(fig)


def hierarchical_clustering(clusters, data, x, y):
      linkage_data = linkage(data, method='ward', metric='euclidean')
      hierarchical_cluster = AgglomerativeClustering(n_clusters=clusters, metric='euclidean', linkage='ward')
      labels = hierarchical_cluster.fit_predict(data)
      # Plot dendogram
      fig, ax = plt.subplots()
      ax.set_title("Dendogram")
      dendrogram(linkage_data)
      st.pyplot(fig)
      # Plot the clusters in a scatter plot
      fig2, ax2 = plt.subplots()
      ax2.set_title("Scatterplot")
      ax2.scatter(x, y, c=labels)
      st.pyplot(fig2)



# General main function of the file should call all the necessary function
interface()




