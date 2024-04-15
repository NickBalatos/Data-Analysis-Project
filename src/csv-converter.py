import pandas as pd
import os

# Take the current directory
current_directory = os.getcwd()

# Generate the path for the wdbc.names and wdbc.data files
names_file_path = os.path.join(current_directory, "wdbc.names")
data_file_path = os.path.join(current_directory, "wdbc.data")

# Read the data from the  wdbc.data file and save them in a DataFrame
data = pd.read_csv("wdbc.data", header=None)

# Read the labels from the wdbc.names file
with open("wdbc.names", "r") as file:
    # Overriding the first 12 lines that contain information about the dataset
    for _ in range(12):
        next(file)
    # Read the next lines that contain the labels
    labels = [line.strip().split(":")[0] for line in file]

# Repeat the labels to match the length of the DataFrame
labels = labels * (len(data) // len(labels)) + labels[:len(data) % len(labels)]

# Add the column names to the DataFrame
data["label"] = labels

# Save the data to a new CSV file
data.to_csv("wdbc_processed.csv", index=False)
