
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston
import seaborn as sns


boston = load_boston()
print("KEYS:", boston.keys())
print(boston.DESCR)
print("Data target:", boston.target)

boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df_no_price = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target

# show all columns in head call
pd.set_option('display.max_columns', None)
print(boston_df.head())
print("MISSING DATA ANALYSIS:\n", boston_df.isnull().sum())

# There is no missing data so we don't need to do any imputing
# Additionally there is no categorical data so there is no need for encoding

print(boston_df.describe())

# Show correlations between non-target features
corr_matrix = boston_df_no_price.corr().round(2)
sns.heatmap(data=corr_matrix, annot=True)

# Print histogram of target feature: price
plt.figure(figsize=(4, 3))
plt.hist(boston.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.tight_layout()

# Scatter plots of price vs value for each feature
for index, feature_name in enumerate(boston.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(boston.data[:, index], boston.target)
    plt.ylabel('Price', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()

# split features and labels into train and test sets, then run linear regression prediction
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=5)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_predicted = lin_reg.predict(X_test)
lin_expected = y_test

# show plot of expected vs predicted values
plt.figure(figsize=(4, 3))
plt.title('Linear Regression')
plt.scatter(lin_expected, lin_predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()

lim_mse = mean_squared_error(lin_expected, lin_predicted)
lin_rmse = np.sqrt(lim_mse)
print("Linear regression RMSE:", lin_rmse)
# Prediction with gradient boosting
gradient_reg = GradientBoostingRegressor()
gradient_reg.fit(X_train, y_train)

grad_predicted = gradient_reg.predict(X_test)
grad_expected = y_test

plt.figure(figsize=(4, 3))
plt.title('Gradient Boosting')
plt.scatter(grad_expected, grad_predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()

grad_mse = mean_squared_error(grad_expected, grad_predicted)
grad_rmse = np.sqrt(grad_mse)
print("Gradient boosting RMSE:", grad_rmse)

# Prediction with random forest regressor
forest_reg = RandomForestRegressor()

forest_expected = y_test

# for random forest use param grid to try different parameter combinations
param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}
    ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final random forest regressor RMSE is", final_rmse)


plt.figure(figsize=(4, 3))
plt.title('Random Forest')
plt.scatter(forest_expected, final_predictions)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()

