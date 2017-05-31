#Testla Stock Prices

# Load libraries
import numpy as np
import quandl, math
import pandas as pd
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


# Load dataset
dataset = quandl.get("WIKI/TSLA", api_key='B2aUqEDSL-Uxhbo1zmug')


# Check dataset
print(dataset.head())
print(dataset.shape)

#dataset.to_csv("datasetRaw.csv", sep=',')


# Prepare dataset
dataset = dataset[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
dataset['HL_PCT'] = (dataset['Adj. High'] - dataset['Adj. Low']) / dataset['Adj. Low'] * 100.0
dataset['OC_PCT'] = (dataset['Adj. Close'] - dataset['Adj. Open']) / dataset['Adj. Open'] * 100.0

dataset = dataset[['Adj. Close', 'HL_PCT', 'OC_PCT', 'Adj. Volume']]

dataset.dropna(inplace=True)

forecast_out = int(math.ceil(0.01*len(dataset)))

dataset['label'] = dataset['Adj. Close'].shift(-forecast_out)

print(dataset)

X = np.array(dataset.drop(['label'], 1))
Y = np.array(dataset['label'])

X = preprocessing.scale(X)

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

X.fillna(X.mean(), inplace=True)
Y.fillna(Y.mean(), inplace=True)

#dataset.to_csv("datasetTrans.csv", sep=',')



# Data visualizations

# histograms
dataset.hist()
pyplot.show()

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
pyplot.show()



# Split-out validation dataset
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)

# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))


# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
models.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
models.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
models.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))


# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Get accuracy best performing algorithms
clf = KNeighborsRegressor()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_validation, Y_validation)

print(accuracy)


clf = KNeighborsRegressor(n_neighbors=21)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_validation, Y_validation)

print(accuracy)

clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_validation, Y_validation)

print(accuracy)

clf = Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor(n_estimators=50))])
clf.fit(X_train, Y_train)
accuracy = clf.score(X_validation, Y_validation)

print(accuracy)