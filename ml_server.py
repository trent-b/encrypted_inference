from Pyfhel import Pyfhel, PyCtxt
from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

# load ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# append model intercept
coefs = np.concatenate((model.coef_, model.intercept_))

@app.route('/predict',methods=['POST'])
def results():
    # create Pyfhel object and load the context and keys sent in the json object
    HE = Pyfhel()
    HE.from_bytes_context(request.json.get('context').encode('cp437'))
    HE.from_bytes_public_key(request.json.get('public_key').encode('cp437'))
    HE.from_bytes_relin_key(request.json.get('relin_key').encode('cp437'))
    HE.from_bytes_rotate_key(request.json.get('rot_key').encode('cp437'))
    
    # get the encrypted data from the json object
    enc_X = PyCtxt(pyfhel=HE, bytestring=request.json.get('enc_X').encode('cp437'))
    
    # convert coefs to Pyfhel plain text object
    pt_coefs = HE.encode(coefs)
    
    # multiply encrypted data with coefficients
    enc_y_predict_orig = enc_X * pt_coefs
    
    # make a copy
    enc_y_predict = PyCtxt(copy_ctxt=enc_y_predict_orig)
    
    # sum result of multiplcation by iteratively shifting
    for idx in range(1,len(coefs)):
        enc_y_predict += enc_y_predict_orig >> idx
    
    # send result in json object
    return {'result':enc_y_predict.to_bytes().decode('cp437')}

if __name__ == "__main__":
    app.run(debug=False)