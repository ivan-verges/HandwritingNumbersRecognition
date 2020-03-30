import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

#Loads the Digits dataset into a Dataframe
digits = datasets.load_digits()

#Creates a KMeans Classifier with a Number of Clusters = 10 and a Random State = 42
model = KMeans(n_clusters = 10, random_state = 42)

#Train the model with the dataset
model.fit(digits.data)

#Creates a new example to predict the numbers
new_samples = np.array([
[0.00,0.00,0.00,3.44,4.35,0.00,0.00,0.00,0.00,0.00,0.00,5.19,6.87,0.00,0.00,0.00,0.00,0.00,0.00,3.82,6.87,0.00,0.00,0.00,0.00,0.00,0.00,3.82,6.87,0.00,0.00,0.00,0.00,0.00,0.00,3.82,6.87,0.00,0.00,0.00,0.00,0.00,0.00,3.82,6.87,0.00,0.00,0.00,0.00,0.00,0.00,3.82,6.87,0.00,0.00,0.00,0.00,0.00,0.00,2.21,4.35,0.00,0.00,0.00],
[0.00,1.45,6.41,7.63,7.63,3.44,0.00,0.00,0.00,4.20,6.87,2.37,6.49,4.58,0.00,0.00,0.00,0.15,0.31,1.07,7.17,4.58,0.00,0.00,0.00,0.00,1.84,6.95,7.56,1.37,0.00,0.00,0.00,0.00,2.60,6.49,7.64,3.37,0.00,0.00,0.00,0.00,0.00,0.08,7.18,4.58,0.00,0.00,0.00,2.29,3.21,5.73,7.63,3.29,0.00,0.00,0.00,6.41,7.63,6.87,3.97,0.08,0.00,0.00],
[0.00,5.04,3.67,0.00,4.43,3.21,0.00,0.00,0.00,6.11,4.58,0.00,6.11,4.58,0.00,0.00,0.00,6.11,4.58,0.00,6.11,4.58,0.00,0.00,0.00,6.11,5.80,3.05,6.72,4.58,0.00,0.00,0.00,5.66,7.64,7.64,7.64,4.58,0.00,0.00,0.00,0.00,0.00,0.00,6.19,4.58,0.00,0.00,0.00,0.00,0.00,0.00,7.64,4.05,0.00,0.00,0.00,0.00,0.00,0.00,7.63,3.05,0.00,0.00],
[0.00,0.00,1.53,5.11,6.10,5.57,1.15,0.00,0.00,0.00,6.57,7.17,5.04,7.55,3.82,0.00,0.00,0.00,7.63,3.74,0.99,7.63,3.82,0.00,0.00,0.00,7.55,5.88,4.96,7.63,3.82,0.00,0.00,0.00,3.13,6.64,6.87,7.56,3.82,0.00,0.00,0.00,0.00,0.00,0.00,6.87,3.82,0.00,0.00,0.00,0.00,0.00,0.00,6.87,3.82,0.00,0.00,0.00,0.00,0.00,0.00,5.72,2.97,0.00]
])

#Makes a prediction for the Samples defined before, then maps the predictions and prints the result
new_labels = model.predict(new_samples)
for label in new_labels:
  if label == 0:
    print(0, end = "")
  elif label == 1:
    print(9, end = "")
  elif label == 2:
    print(2, end = "")
  elif label == 3:
    print(1, end = "")
  elif label == 4:
    print(6, end = "")
  elif label == 5:
    print(8, end = "")
  elif label == 6:
    print(4, end = "")
  elif label == 7:
    print(5, end = "")
  elif label == 8:
    print(7, end = "")
  elif label == 9:
    print(3, end = "")

#Prepare a Plot to show the labels to make predictions
fig = plt.figure(figsize = (8, 3))

#Set a title for the graph, font size and style
fig.suptitle("Cluster Center Images", fontsize = 14, fontweight = "bold")

#Loops from 0 to 9 and plots the graph for each labl
for i in range(10):
  ax = fig.add_subplot(2, 5, 1 + i)
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap = plt.cm.binary)

#Show the graph
plt.show()