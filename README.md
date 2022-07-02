# Homomorphic Encryption in Machine Learning as a Service (MLaaS)
Example of homomorphic encryption in machine learning inference as a service.

### Train Model
`train_model.py` trains a regularized linear regression model on the California Housing Dataset, which is public data not requiring encryption but is nonetheless used here to demonstrate proof of concept.

### Start Server
`ml_server.py` is a Flask application for the REST API. It accepts encrypted data from the client and returns a prediction without ever decrypting the data.

### Run Client
`client.py` sends encrypted data to the server application and receives a response with a prediction that it decrypts. 


