import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

 #categories = [
#     'alt.atheism',
#     'talk.religion.misc',
#     'comp.graphics',
#     'sci.space'
#]

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

## KMEANS CLUSTERING
dataset = fetch_20newsgroups(subset='all',
                             shuffle=True, random_state=42)
categories = dataset.target_names


print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print(dataset.target_names)
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(dataset.data)
k_values = range(1, 61)
inertia_values = [MiniBatchKMeans(k).fit(X).inertia_
                  for k in k_values]
plt.plot(k_values, inertia_values)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.axvline(20, c='k')
plt.grid(True)
plt.show()


km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(dataset.data)
print("Clustering sparse data with %s" % km)
km.fit(X)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names_out()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
        print()

## NAIVE BAYES ##
print("### NAIVE BAYES ###")
data_train = fetch_20newsgroups(subset='train', 
                                categories=categories, 
                                shuffle=True, random_state=42)
n_components = 5
labels = data_train.target
true_k = np.unique(labels).shape[0]

# Convert to TF-IDF format
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)
X_train = vectorizer.fit_transform(data_train.data)

print(len(data_train.data))

data_test = fetch_20newsgroups(subset='test', 
                               categories=categories, 
                               shuffle=True, random_state=42)

target_names = data_train.target_names

# Split a train set and test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(data_test.data)


nb_clf = GaussianNB()
nb_clf.fit(X_train.toarray(), y_train)

nb_pred = nb_clf.predict(X_train.toarray())
train_score = accuracy_score(y_train, nb_pred) * 100
print(f"Train accuracy score: {train_score:.2f}%")

nb_pred = nb_clf.predict(X_test.toarray())
test_score = accuracy_score(y_test, nb_pred) * 100
print(f"Test accuracy score: {test_score:.2f}%")

cm = confusion_matrix(y_test, nb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=data_train.target_names)


fig, ax = plt.subplots(figsize=(10, 10))
disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='winter')

plt.show()

print(pd.DataFrame(classification_report(y_test, nb_pred, output_dict=True)).T)