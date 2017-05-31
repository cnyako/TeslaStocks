# Save Model Using joblib
import numpy as np
import quandl, math
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = quandl.get("WIKI/TSLA", api_key='B2aUqEDSL-Uxhbo1zmug')

# Prepare dataset
dataset = dataset[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
dataset['HL_PCT'] = (dataset['Adj. High'] - dataset['Adj. Low']) / dataset['Adj. Low'] * 100.0
dataset['OC_PCT'] = (dataset['Adj. Close'] - dataset['Adj. Open']) / dataset['Adj. Open'] * 100.0

dataset = dataset[['Adj. Close', 'HL_PCT', 'OC_PCT', 'Adj. Volume']]

dataset.dropna(inplace=True)

forecast_out = int(math.ceil(0.01*len(dataset)))

dataset['label'] = dataset['Adj. Close'].shift(-forecast_out)

X = np.array(dataset.drop(['label'], 1))
Y = np.array(dataset['label'])

X = preprocessing.scale(X)

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

X.fillna(X.mean(), inplace=True)
Y.fillna(Y.mean(), inplace=True)

validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

grad = Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor(n_estimators=50))])
grad.fit(X_train, Y_train)
accuracy = grad.score(X_validation, Y_validation)

print(accuracy)

# save the model to disk
filename = 'Stocks_finalized_model.sav'
dump(grad, filename)
