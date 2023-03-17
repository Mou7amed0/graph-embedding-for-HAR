

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


DATA_PATH = 'C:\\Users\\Mohamed\\PycharmProjects\\DATA'

actions = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping',
           'drinking','opening microwave oven', 'Throwing trash']



labels = np.load(DATA_PATH + "/Train/y_train.npy")
X = np.load("embeddings/train/train_embedding_dim8.npy")

length = len(labels)

labs, counts = np.unique(labels, return_counts=True)
X_flat = X.reshape((length, -1))
X_train, X_test, y_train, y_test = train_test_split(X_flat, labels, test_size=0.33, random_state=111)

from sklearn.svm import SVC
cls = SVC(kernel="linear")
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))