# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets


from sklearn.datasets import load_diabetes
df = load_diabetes()
df.keys()
unscaled_X = df.data
scaler = StandardScaler()

X =scaler.fit_transform(unscaled_X)

y = df.target


# Compute paths

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


#Lasso Regression using grid search cross validation

LassoRegression = Lasso()
model_LassoRegression = LassoRegression.fit(X_train, y_train)
y_predictLasso_training = LassoRegression.predict(X_train)
y_predictLasso_testing = LassoRegression.predict(X_test)

print('Training Accuracy - R2 Score Lasso Regression:', r2_score(y_train, y_predictLasso_training))
print('Testing Accuracy - R2 Score Lass oRegression:', r2_score(y_test, y_predictLasso_testing))

MeanSquaredErrorLasso = cross_val_score(LassoRegression, X, y, scoring = 'neg_mean_squared_error', cv=5)
mean_MeanSquaredErrorLasso = np.mean(MeanSquaredErrorLasso)
print(-(mean_MeanSquaredErrorLasso))

FeatureCoefficientsLasso = pd.Series(LassoRegression.coef_, index = df.feature_names)
FeatureCoefficientsLasso.plot(kind = 'barh')
plt.title('Coefficients of Each Feature Using Lasso Regression')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()

Lasso2 = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.9, 0,8, 0.6, 0.5, 0.4, 1, 2, 5, 10,20,25, 30, 35, 40, 50, 55, 75, 100 ]}
lasso_regressor = GridSearchCV(Lasso2, parameters, scoring = 'neg_mean_squared_error', cv = 10)
lasso_regressor.fit(X_train, y_train)

print('Best Parameter Value Lasso Regression:', lasso_regressor.best_params_)
print('Lasso Best MSE:', -(lasso_regressor.best_score_))

y_predictLasso2_training = lasso_regressor.predict(X_train)
y_predictLasso2_testing = lasso_regressor.predict(X_test)

print('Training Accuracy - R2 Score Lasso Regression:', r2_score(y_train, y_predictLasso2_training))
print('Testing Accuracy - R2 Score Lasso Regression:', r2_score(y_test, y_predictLasso2_testing))


#elastic net regression using cross validation
enet = ElasticNet()
model_enet = enet.fit(X_train, y_train)
y_predenet_train = enet.predict(X_train)
y_predenet_test = enet.predict(X_test)



enet2 = ElasticNet()
parameters = {'alpha': [1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.9, 0,8, 0.6, 0.5, 0.4, 1, 2, 5, 10,20,25, 30, 35, 40, 50, 55, 75, 100 ]}
enet_regressor = GridSearchCV(enet2, parameters, scoring = 'neg_mean_squared_error', cv = 10)
enet_regressor.fit(X_train, y_train)


print('Training Accuracy - R2 Score Elastic Net Regression:', r2_score(y_train, y_predenet_train))
print('Testing Accuracy - R2 Score Elastic Net Regression:', r2_score(y_test, y_predenet_test))

print('Best Parameter Value Elastic Net Regression:', enet_regressor.best_params_)
print('Elastic Net Best MSE:', -(enet_regressor.best_score_))

y_predictenet_training = enet_regressor.predict(X_train)
y_predictenet_testing = enet_regressor.predict(X_test)


FeatureCoefficientsElasticNet = pd.Series(enet.coef_, index = df.feature_names)
FeatureCoefficientsElasticNet.plot(kind = 'barh')
plt.title('Coefficients of Each Feature Using Elastic Net')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()

#regularization paths
eps = 5e-8  # the smaller it is the longer is the path

alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps)
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(X, y, eps=eps, positive=True)
alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8)

alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X_train, y_train, eps=eps, l1_ratio=0.8, positive=True)

# Display results

plt.figure(1)
colors = cycle(["b", "r", "g", "c", "k"])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle="--", c=c)

plt.xlabel("-Log(alpha)")
plt.ylabel("Feature Coefficients")
plt.title("Lasso and Elastic-Net Regularization Paths")
plt.legend((l1[-1], l2[-1]), ("Lasso", "Elastic-Net"), loc="lower left")
plt.axis("tight")

plt.show()
