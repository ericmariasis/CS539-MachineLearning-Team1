import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import density
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import l1_min_c
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import enet_path
from itertools import cycle
import pylab as pl

dataset = fetch_20newsgroups(subset='all',
                             shuffle=True, random_state=42)

categories = dataset.target_names

print(dataset.target_names)
print()

labels = dataset.target

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)

y_train, y_test = data_train.target, data_test.target

# Turn down for faster run time
#n_samples = 250

X, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True, as_frame=True)
#X = X[:n_samples]
#y = y[:n_samples]
X = SelectKBest(chi2, k=20).fit_transform(X, y)

y_train_vec, y_test_vec = y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.1
)

# ## ALTERNATE VECTORIZATION
# vectorizer = CountVectorizer()
# X_train_vec = vectorizer.fit_transform(data_train.data)
# X_test_vec = vectorizer.transform(data_test.data)

# feature_names = vectorizer.get_feature_names_out()
# ch2 = SelectKBest(chi2, k=20)
# X_train_vec = ch2.fit_transform(X_train_vec, y_train_vec)
# X_test_vec = ch2.transform(X_test_vec)
# if feature_names is not None:
#     # keep selected feature names
#     feature_names = feature_names[ch2.get_support()]

#print("FEATURE NAMES ARE", feature_names)

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))


        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

cs = l1_min_c(X, y, loss="log") * np.logspace(0, 7, 16)

clf = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    tol=1e-6,
)
benchmark(clf)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X_train, y_train)
    coefs_.append(clf.coef_.ravel().copy())

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker="o")
ymin, ymax = plt.ylim()
plt.xlabel("log(C)")
plt.ylabel("Coefficients")
plt.title("Logistic Regression L1 Path")
plt.axis("tight")
plt.show()

# L2 regularization
clf = LogisticRegression(
    tol=1e-6,
)
benchmark(clf)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X_train, y_train)
    coefs_.append(clf.coef_.ravel().copy())

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker="o")
ymin, ymax = plt.ylim()
plt.xlabel("log(C)")
plt.ylabel("Coefficients")
plt.title("Logistic Regression L2 Path")
plt.axis("tight")
plt.show()

# #Elastic Net
# model_enet = ElasticNet(alpha = 0.01)
# model_enet.fit(X_train, y_train) 
# pred_train_enet= model_enet.predict(X_train)
# print("RMSE train", np.sqrt(mean_squared_error(y_train,pred_train_enet)))
# print("r2 score train", r2_score(y_train, pred_train_enet))

# pred_test_enet= model_enet.predict(X_test)
# print("RMSE test", np.sqrt(mean_squared_error(y_test,pred_test_enet)))
# print("r2 score test", r2_score(y_test, pred_test_enet))



# ##############################################

# alphas_enet, coefs_enet, _ = enet_path(X, y, eps=5e-3, l1_ratio=0.8)



# # plt.figure(1)
# colors = cycle(["b", "r", "g", "c", "k"])
# # for coef_e, c in zip(coefs_enet, colors):
# #     plt.plot(alphas_enet, coef_e, linestyle="--", c=c)
# # ymin, ymax = plt.ylim()
# # plt.xlabel("log(C)")
# # plt.ylabel("Coefficients")
# # plt.title("Elasticnet")
# # plt.axis("tight")
# # plt.show()

# print("Computing regularization path using the elastic net...")
# # models = enet_path(X, y, eps=5e-3, l1_ratio=0.8)
# # alphas_enet = np.array([model.alpha for model in models])
# # coefs_enet = np.array([model.coef_ for model in models])

# ###############################################################################
# # Display results

# plt.figure(1)
# colors = cycle(["b", "r", "g", "c", "k"])
# neg_log_alphas_enet = -np.log10(alphas_enet)
# for coef_e, c in zip(coefs_enet, colors):
#     l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle="--", c=c)

# plt.xlim(-3,4)
# plt.xlabel("-Log(alpha)")
# plt.ylabel("coefficients")
# plt.title("Lasso and Elastic-Net Paths")
# plt.axis("tight")
# plt.show()






clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.8,
        tol=1e-6,
)
benchmark(clf)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X_train, y_train)
    coefs_.append(clf.coef_.ravel().copy())

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker="o")
ymin, ymax = plt.ylim()
plt.xlabel("log(C)")
plt.ylabel("Coefficients")
plt.title("Logistic Regression ElasticNet")
plt.axis("tight")
plt.show()

