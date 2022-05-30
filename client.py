from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from Pyfhel import Pyfhel, PyCtxt
import requests
import pickle

url = 'http://localhost:5000/predict'

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X = X.values
y = y.values.reshape(-1,1)

_, X, _, y = train_test_split(X, y, test_size=0.1, random_state=0)

with open('ss.pkl', 'rb') as f:
    ss = pickle.load(f)

X = ss.fit_transform(X)

HE = Pyfhel(context_params={'scheme':'ckks', 'n':2**13, 'scale':2**30, 'qi':[30]*5})
HE.keyGen()
HE.relinKeyGen()
HE.rotateKeyGen()

s_context = HE.to_bytes_context().decode('cp437')
s_relin_key = HE.to_bytes_relin_key().decode('cp437')
s_rotate_key = HE.to_bytes_rotate_key().decode('cp437')
s_public_key = HE.to_bytes_public_key().decode('cp437')

n_samples = 10
y_predict = []

for idx in range(n_samples):
    X_row = X[idx]
    X_row = np.concatenate((X_row, [1]))
    
    enc_X = HE.encrypt(X_row)
    s_enc_X = enc_X.to_bytes().decode('cp437')
    
    r = requests.post(url,json={'context':s_context, 'public_key':s_public_key, 
                                'relin_key':s_relin_key, 'rot_key':s_rotate_key,
                                'enc_X':s_enc_X})
    
    c_res = PyCtxt(pyfhel=HE, bytestring=r.json()['result'].encode('cp437'))
    res = HE.decryptFrac(c_res)
    pred = res[X.shape[1]]
    print(f'${int(1E5 * pred):,}')
    y_predict.append(pred)

mae = 1E5 * mean_absolute_error(y[:n_samples], y_predict)
print(f'\nMAE = ${int(mae):,}')
