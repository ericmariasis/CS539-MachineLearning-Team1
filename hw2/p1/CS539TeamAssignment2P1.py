
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn import metrics
import sklearn.datasets as ds
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#visualizing the dataset
data, labels = ds.make_circles(n_samples=1000, factor = 0.5,
                             shuffle=True,
                             noise=0.05,
                             random_state=None)

data += np.array(-np.ndarray.min(data[:,0]), 
                 -np.ndarray.min(data[:,1]))

np.ndarray.min(data[:,0]), np.ndarray.min(data[:,1])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.scatter(data[labels==0, 0], data[labels==0, 1], 
               c='pink', s=30, label='pink')
ax.scatter(data[labels==1, 0], data[labels==1, 1], 
               c='blue', s=30, label='blues')

ax.set(xlabel='X',
       ylabel='Y',
       title='Visualization of the Make Circles Dataset: N=1000') #this was repeated but for make_moons


plt.show()


np.random.seed(0)
n_samples = 1000
circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
moons = datasets.make_moons(n_samples=n_samples, noise=.05)


plt.figure(figsize=(13.7, 14.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'n_neighbors': 2,
                'n_clusters': 2}

datasets = [
    (circles, {'n_clusters': 2}),
    (moons, {'n_clusters': 2}),
]


for i_dataset, (dataset, algo_params) in enumerate(datasets):
    parameters = default_base.copy()
    parameters.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # ============
    # Create cluster objects
    # ============

    complete = cluster.AgglomerativeClustering(
        n_clusters=parameters['n_clusters'], linkage='complete')
    average = cluster.AgglomerativeClustering(
        n_clusters=parameters['n_clusters'], linkage='average')
    single = cluster.AgglomerativeClustering(
        n_clusters=parameters['n_clusters'], linkage='single')

    clustering_algorithms = (
        ('Single Linkage', single),
        ('Average Linkage', average),
        ('Complete Linkage', complete))

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_predict = algorithm.labels_.astype(int)
        else:
            y_predict = algorithm.predict(X)

            

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)


        colors = np.array(list(islice(cycle(['Blue', 'Pink', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_predict) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predict])

        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
plt.title('Hierarchical Clustering')
plt.show()

from sklearn.metrics.cluster import homogeneity_score, completeness_score


Homo = homogeneity_score(labels_true= y, labels_pred=y_predict)
print('Homogeinity Score:\n', Homo)

Comp = completeness_score(labels_true=y, labels_pred= y_predict)
print('Completeness Score:\n', Comp)



datapoints, labels = make_moons(noise=0.05, random_state=0) #changed this to make_circles and added factor = 0.5 to run that
(nsamples, nfeatures), n_digits = datapoints.shape, np.unique(labels).size
print("Data Shape", datapoints.shape)
print(f"# digits: {n_digits}; # samples: {nsamples}; # features {nfeatures}")
cluster_count = 2
kmeans = KMeans(n_clusters=cluster_count)
kmeans.fit(datapoints)
clusters = kmeans.predict(datapoints)


def bench_k_means(clusters, name, datapoints, labels):
    
    t0 = time()
    estimator = make_pipeline(StandardScaler(), clusters).fit(datapoints)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            datapoints,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=1000,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t\t\t{:.3f}s\t\t{:.0f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}"
    )
    print(formatter_result.format(*results))


print(82 * "_")
print("Initialization\t\tTime\tInertia\tHomogeinity\tCompleteness\tSilhouette Coefficient")

kmeans = cluster.KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(clusters=kmeans, name="k-means++", datapoints=datapoints, labels=labels)

kmeans = cluster.KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(clusters=kmeans, name="random", datapoints=datapoints, labels=labels)

#pca = PCA(n_components=n_digits).fit(datapoints)
#kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
#bench_k_means(clusters=kmeans, name="PCA-based", datapoints=datapoints, labels=labels)

print(82 * "_")

reduced_data = PCA(n_components=2).fit_transform(datapoints)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

h = 0.02 
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap = plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=165,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on make_moons toy dataset \n"
    "Centroids shown as White X's"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show() #this was repeated for make_circles dataset