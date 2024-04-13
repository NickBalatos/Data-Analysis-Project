import streamlit as st
from st_pages import show_pages_from_config, add_page_title
import numpy as np# for testing reasons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

def interface():
      st.title("K-Means")
      st.write(
            """
            Ο αλγόριθμος K-Means είναι ένας clustering αλγόριθμος μηχανικής μάθησης που χρησιμοποιείται στη στατιστική ανάλυση για την ομαδοποίηση δεδομένων σε έναν προκαθορισμένο αριθμό ομάδων, k.
            Ο αλγόριθμος επιλέγει τυχαία k κέντρα και στη συνέχεια αναθέτει κάθε σημείο δεδομένων στην πλησιέστερη ομάδα, βελτιστοποιώντας τη θέση των κέντρων μέχρι να σταθεροποιηθούν.
            Χρησιμοποιείται ευρέως για την ανίχνευση μοτίβων, την ανάλυση ομάδων και ως προπαρασκευαστικό βήμα για άλλες αλγοριθμικές εφαρμογές.
            """
      )
      st.number_input(label="Αριθμός ομάδων", min_value=1, max_value=5)
      if st.button("Run", key="kmeans"):
            st.write("Run K-Means")
            # k_means(clusters)

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

def k_means(clusters):
      x = np.random.uniform(1, 10, 100) # rows from the csv file
      y = np.random.uniform(1, 10, 100) # the target 
            
      data = list(zip(x, y))

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

interface()
k_means(2)




