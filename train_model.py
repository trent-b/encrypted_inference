from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X = X.values
y = y.values.reshape(-1,1)

# load training data
X, _, y, _ = train_test_split(X, y, test_size=0.1, random_state=0)

model = ElasticNet(alpha=0.001, l1_ratio=0.05)

kf = KFold(n_splits=10, shuffle=True, random_state=1)
ss = StandardScaler()

scores = []

# K-fold cross-validation
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    scores.append(1E5 * mean_absolute_error(y_predict, y_test))

# fit StandardScaler
X = ss.fit_transform(X)

# save StandardScaler
with open('ss.pkl', 'wb') as f:
    pickle.dump(ss, f)

# retrain model with all training data
model.fit(X, y)

# save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f'Mean MAE (SD) = ${int(np.mean(scores)):,} (${int(np.std(scores)):,})')
