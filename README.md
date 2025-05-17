# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries and load data.

2.Select features for clustering.

3.Fit KMeans model with chosen clusters (e.g., k=5).

4.Predict clusters and plot results.

5.Print cluster centers and finish.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: BHUVANESH K
RegisterNumber:  212224230037
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

data = pd.read_csv(io.BytesIO(uploaded['Mallcustomers.csv']))
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])
plt.xlabel('Income')
plt.ylabel('Score')
plt.show()
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
colors = ['r', 'g', 'b', 'c', 'm']
for i in range(5):
    plt.scatter(X[labels==i]['Annual Income (k$)'], X[labels==i]['Spending Score (1-100)'], color=colors[i])
plt.scatter(centers[:,0], centers[:,1], color='black', s=200, label='Centroids')
plt.xlabel('Income')
plt.ylabel('Score')
plt.legend()
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/cbf7db01-1515-4bc1-bd5a-6479f5ddceb6)


![image](https://github.com/user-attachments/assets/a4fe69d8-940b-4784-b632-371ca3a94178)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
