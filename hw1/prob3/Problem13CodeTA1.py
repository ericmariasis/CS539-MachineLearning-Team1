from ast import increment_lineno
import matplotlib
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import datasets, linear_model,svm
from sklearn.linear_model import LinearRegression, ridge_regression, BayesianRidge
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

#importing the dataset
diabetes = datasets.load_diabetes()

#dataset descriptives and printing the features and shape of data (X) and the target
print(diabetes.DESCR)

diabetes.feature_names
print('Data set feature names: \n' , diabetes.feature_names)

diabetes.data.shape
print('Diabetes Data Shape \n', diabetes.data.shape)

diabetes.target.shape
print('Diabetes Target shape \n', diabetes.target.shape)

#making dataframe
diabetes_df = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
diabetes_df['Disease Progression'] = diabetes.target
diabetes_df.describe()

#seeing the corrrelation matrix
corr = diabetes_df.corr()
print(corr)

plt.subplots(figsize=(10,10))
sns.heatmap(corr,cmap = 'YlGnBu')
plt.show()

#defining X and y from the datafram
diabetes_X = diabetes_df.drop(labels = 'Disease Progression', axis=1)
diabetes_y = diabetes_df['Disease Progression']

#splitting into X and Y testing and trianing samples
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y, test_size = 0.2, random_state = 999)

#Linear regression model for disease progression
LR = linear_model.LinearRegression()

#fitting the training data
LR.fit(diabetes_X_train, diabetes_y_train)

#using the linear regression to predict Y given X test data
diabetes_progression_pred = LR.predict(diabetes_X_test)
intercept = LR.intercept_

# The regression coefficients, MSE and R2 - I cant get this to print anymore for some reason ever since I switched to using the data in a dataframe
print('Regression Coefficients: \n', LR.coef_)
print('Intercept \n', LR.intercept_)
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, diabetes_progression_pred))
print('R2 Score: %.2f' % r2_score(diabetes_y_test, diabetes_progression_pred))


#also we are supposed to use cross validation but I cant get it to print the scores :(
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, diabetes_X, diabetes_y, cv=5)
print('Cross Validation Scores: %0.2f accuracy with standard deviation of %0.2f \n' %(scores.mean(), scores.std()))


#Ridge Regression

#fitting the training data using ridge regression
RR = linear_model.ridge_regression
clf = RR.fit(diabetes_X_train,diabetes_y_train)

y_predict = clf.predict(diabetes_X_test)

clf.score(diabetes_X_test, y_predict)

print("Clf Score:", clf.score(diabetes_X_test,y_predict))

#BayesianRidgeRegression
np.random.seed(0)
n_samples, n_features = 100,100

X = np.random.randn(n_samples, n_features)
lambda_ =4
w= np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. /np.sqrt(lambda_))

    alpha_=50.
    noise = stats.norm.rvs(loc=0, scale =1 / np.sqrt(alpha_), size = n_samples)
    y = np.dot(X, w) + noise

clf = BayesianRidge(compute_score=True)
clf.fit(diabetes_X_train, diabetes_y_train)

y_predict = clf.predict(diabetes_X_test)

clf.score(diabetes_X_test, y_predict)

print("Clf Score:", clf.score(diabetes_X_test,y_predict))

#KNN Classifier

KNN = KNeighborsClassifier(n_neighbors=10)

#fitting the model
KNN.fit(diabetes_X_train, diabetes_y_train)

KNN.predict(diabetes_X_test) [0:5]
KNN.score(diabetes_X_test,diabetes_y_test)
print('KNN Classifier Score: \n', KNN.score)

KNN_CV = KNeighborsClassifier(n_neighbors=10)
CV_scores = cross_val_score(KNN_CV, diabetes_X, diabetes_y, cv=5)
print(CV_scores)
print('CV_Scores: []'.format(np.mean(CV_scores)))
