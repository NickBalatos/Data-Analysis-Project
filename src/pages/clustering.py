import streamlit as st
from st_pages import show_pages_from_config, add_page_title
import numpy as np# for testing reasons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

def get_data(x, y, data):
      # It should fetch the csv data from the home page
      # because the home page isn't ready, it will generate the required data
      local_x = np.random.uniform(1, 10, 100) # rows from the csv file
      local_y = np.random.uniform(1, 10, 100) # the target 

      
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

      st.title("Second clustering algorithm")
      st.write(
            """
            Lorem ispum bla bla bla
            """
      )
      st.number_input(label="Algorithm parameter", min_value=1, max_value=5)
      if st.button("Run", key="secalgo"):
            st.write("Run the second algorithm")
            # funtion_sec_algorithm()


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
      st.pyplot(fig) 
      # arr = np.random.normal(1, 1, size=100)
      # fig, ax = plt.subplots()
      # ax.hist(arr, bins=20)

      # st.pyplot(fig)


# General main function of the file should call all the necessary function
interface()




