import matplotlib
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report, accuracy_score
from sklearn import datasets, linear_model,svm
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
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

#Corrrelation matrix
corr = diabetes_df.corr()
print(corr)
plt.subplots(figsize=(10,10))
plt.title("Correlation Matrix")
sns.heatmap(corr,cmap = 'YlGnBu')
#plt.show()

#defining X and y from the dataframe
diabetes_X = diabetes_df.drop(labels = 'Disease Progression' ,axis=1)
diabetes_y = diabetes_df['Disease Progression']

#splitting into X and Y testing and trianing samples
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y, test_size = 0.25, random_state = 1)

#Linear regression model for disease progression
LR = linear_model.LinearRegression()

#fitting the training data
regressionline = LR.fit(diabetes_X_train, diabetes_y_train)

#using the linear regression to predict Y given X test data
diabetes_progression_pred1 = LR.predict(diabetes_X_test)

intercept = LR.intercept_

# The regression coefficients, MSE and R2 
print('Linear Regression Coefficients: \n', LR.coef_)
print('Intercept: \n', LR.intercept_)
print('Mean squared error:  %.2f' % mean_squared_error(diabetes_y_test, diabetes_progression_pred1))

LRR2scoreTest = r2_score(diabetes_y_test, diabetes_progression_pred1)
print('R2 Score Linear Regression on Testing Data: \n %.2f ' %LRR2scoreTest)

CVScoreLR = cross_val_score(LR, diabetes_X_test, diabetes_y_test, cv=3)
print("Cross Validation Score for Linear Regression: \n", CVScoreLR)
AccuracyLR = average(CVScoreLR)
print("Accuracy of LR: \n", AccuracyLR)
#Ridge Regression



#fitting the training data using ridge regression
n_samples, n_features = 442,10 
clf2 = Ridge(alpha=1.0)
clf2.fit(diabetes_X_train,diabetes_y_train)
diabetes_progression_pred2 = clf2.predict(diabetes_X_test)
clf2.score(diabetes_X_test, diabetes_progression_pred2)


print('Ridge Regression Coefficients: \n', clf2.coef_)
print('Ridge Regressor Intercept: \n', clf2.intercept_)
print('Mean squared error of Ridge Regression:  %.2f' % mean_squared_error(diabetes_y_test, diabetes_progression_pred2))
print('R2 Score: \n %.2f ' %r2_score(diabetes_y_test, diabetes_progression_pred2))

print('Cross Validation Score for Ridge Regression: \n', cross_val_score(clf2, diabetes_X, diabetes_y, cv=3))
AccuracyRR = average(cross_val_score(clf2, diabetes_X, diabetes_y, cv=3))

print('Accuracy of Ridge Regression: \n', AccuracyRR)
#BayesianRidgeRegression
np.random.seed(0)
n_samples, n_features = 442,10

X = np.random.randn(n_samples, n_features)
lambda_ =4
w= np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. /np.sqrt(lambda_))

    alpha_=50.
    noise = stats.norm.rvs(loc=0, scale =1 / np.sqrt(alpha_), size = n_samples)
    y = np.dot(X, w) + noise

clf3 = BayesianRidge(compute_score=True)
clf3.fit(diabetes_X_train, diabetes_y_train)

y_predict = clf3.predict(diabetes_X_test)

clf3.score(diabetes_X_test, y_predict)
print('Bayesian Ridge Coefficients: \n', clf3.coef_)
print('Bayesian Ridge Regressor Intercept: \n', clf3.intercept_)
print('Mean squared error Bayesian Regression:  %.2f' % mean_squared_error(diabetes_y_test, y_predict))
print("R2 Score Bayesian Ridge Regression: \n %0.2f" %r2_score(diabetes_y_test, y_predict))
print("Cross Validation Score for Bayesian Regression: \n", cross_val_score(clf3, diabetes_X, diabetes_y, cv=10))
AccuracyBRR = average(cross_val_score(clf3, diabetes_X, diabetes_y, cv=3))

print('Accuracy of Bayesian Ridge Regression: \n', AccuracyBRR)


#KNN Classifier

KNN = KNeighborsClassifier(n_neighbors=15)

#fitting the model
KNN.fit(diabetes_X_train, diabetes_y_train)

prediction = KNN.predict(diabetes_X_test)
error_rate = []

KNN.score(diabetes_X_test,diabetes_y_test)
print('Mean Squared Error KNN:\n', mean_squared_error(diabetes_y_test, prediction))
print('KNN Regressor R2 Score:\n', r2_score(diabetes_y_test, prediction))

KNN_CV = KNeighborsClassifier(n_neighbors=3)
CVscore = cross_val_score(KNN, diabetes_X, diabetes_y, cv=5)
print('Cross Validation Score for KNN: \n',CVscore)

AccuracyKNN = average(cross_val_score(KNN, diabetes_X, diabetes_y, cv=3))

print('Accuracy of KNN Regression: \n', AccuracyKNN)