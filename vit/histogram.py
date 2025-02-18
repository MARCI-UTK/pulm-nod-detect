import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

root_path = '/data/marci/luna16'

img_metadata_paths = [os.path.join(root_path, 'processed_scan', f) for f in os.listdir(os.path.join(root_path, 'processed_scan'))]

img_paths = [p for p in img_metadata_paths if p.endswith('.npy')]
metadata_paths = [p for p in img_metadata_paths if p.endswith('.json')]

annotations_path = os.path.join(root_path, 'csv', 'annotations.csv')
annotations = pd.read_csv(annotations_path)

sizes = []
small_nods = []
for ip, mp in zip(img_paths, metadata_paths): 
    id = ip.split('/')[-1][0:-4]

    locs = annotations[annotations['seriesuid'] == id]

    if len(locs) == 0: 
        continue

    for _, l in locs.iterrows(): 
        size = l['diameter_mm']
        sizes.append(size)

        if size <= 12.93: 
            small_nods.append(size)

num_bins = 3
counts, bin_edges = np.histogram(sizes, bins=num_bins)

# Create the histogram
plt.figure(figsize=(8, 6))
plt.hist(sizes, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')

# Format X-axis to show bin edges
bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
plt.xticks(bin_edges, rotation=45, ha='right') 

# Add labels and title
plt.xlabel("Nodule Size")
plt.ylabel("Frequency")
plt.title("Histogram of Nodule Sizes")

# Show grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the histogram
plt.savefig('all_hist.png')

# Clear plot 
plt.cla()

# Plot small nodule histogram
num_bins = 5
counts, bin_edges = np.histogram(small_nods, bins=num_bins)

# Create the histogram
plt.figure(figsize=(8, 6))
plt.hist(small_nods, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')

# Format X-axis to show bin edges
bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
plt.xticks(bin_edges, rotation=45, ha='right') 

# Add labels and title
plt.xlabel("Nodule Size")
plt.ylabel("Frequency")
plt.title("Histogram of Nodule Sizes <= 12.93mm")

# Show grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the histogram
plt.savefig('small_hist.png')