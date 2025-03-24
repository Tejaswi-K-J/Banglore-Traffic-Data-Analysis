import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skfuzzy.cluster import cmeans

# Load dataset
data = pd.read_csv('Bangalore_Traffic_Dataset_With_Time.csv')

# Check if "Time of Day" column exists
if "Time of Day" not in data.columns:
    raise ValueError("The dataset is missing a 'Time of Day' column!")

# Encode categorical column (Area Name)
encoder = LabelEncoder()
data['Area Name Encoded'] = encoder.fit_transform(data['Area Name'])

# Convert relevant columns to numeric
numeric_columns = ["Traffic Volume", "Average Speed", "Congestion Level"]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop any remaining non-numeric values
data.dropna(inplace=True)

# Define clustering parameters
n_clusters = 3
m = 1.5
max_iter = 1000
error = 0.005

# Set up 4 subplots for different time intervals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
time_intervals = data["Time of Day"].unique()

# Dictionary to store clustered results
cluster_results = {}

# Perform clustering for each time interval and plot
for idx, time in enumerate(time_intervals):
    subset = data[data["Time of Day"] == time].copy()

    # Ensure subset is not empty
    if subset.empty:
        print(f"Skipping clustering for {time} (No data available)")
        continue

    # Select relevant features
    selected_features = ["Area Name Encoded", "Traffic Volume", "Average Speed", "Congestion Level"]
    X = subset[selected_features]

    # Standardize the numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)

    # Apply Fuzzy C-Means clustering
    cntr, u, u0, d, jm, p, fpc = cmeans(
        scaled_features.T, n_clusters, m, error=error, maxiter=max_iter, init=None
    )

    # Assign cluster labels
    subset["Cluster"] = np.argmax(u, axis=0)

    # Map encoded area names back to actual names
    subset["Area Name"] = encoder.inverse_transform(subset["Area Name Encoded"].astype(int))

    # Store results for later summary
    cluster_results[time] = subset.copy()

    # Scatter plot visualization in subplots
    ax = axes[idx // 2, idx % 2]
    for i in range(n_clusters):
        ax.scatter(subset["Traffic Volume"][subset["Cluster"] == i],
                   subset["Average Speed"][subset["Cluster"] == i],
                   label=f'Cluster {i}')
    ax.set_xlabel("Traffic Volume")
    ax.set_ylabel("Average Speed")
    ax.set_title(f"Traffic Clustering - {time}")
    ax.legend()

plt.tight_layout()
plt.show()

# Display cluster-wise summary for each time period (Including all Area Names in each cluster)
for time, df in cluster_results.items():
    print(f"\nCluster Summary for {time}:")
    summary = df.groupby("Cluster").agg({
        "Traffic Volume": "mean",
        "Average Speed": "mean",
        "Congestion Level": "mean",
        "Area Name": lambda x: ", ".join(x.unique())  # List all unique area names in the cluster
    }).reset_index()
    print(summary)
