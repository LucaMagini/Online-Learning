import flask
from flask_selfdoc import Autodoc
from flask import request

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False
auto = Autodoc(app)

ip_address = '0.0.0.0'


@app.route('/', methods = ['GET'])
@auto.doc()
def home():
    "Checking Server Operation"
    
    try:
        result = {
                    'Result':"OK",
                    'Data':"Server Ready"
                 }
            
    except:
        result = {
                    'Result':"NOT OK",
                    'Data':"Server Not Ready"
                 }
            
    response = flask.jsonify(result)
    response.headers.set('Content-Type', 'application/json')
    
    return response  

@app.route('/api/v1/create_model')
@auto.doc(args=['model', 'name', 'intercept_lr', 'loss', 'n_models', 
                'split_criterion', 'n_neighbors', 'p'])
def create_model():
    "Create a ML Model for Online Learning"
    
    models = ['logisticregression', 'randomforest', 'knn', 'gaussiannb']
    lr_params = ['model', 'name', 'intercept_lr', 'loss']
    rf_params = ['model', 'name', 'n_models', 'split_criterion']
    knn_params = ['model', 'name', 'n_neighbors', 'p']
    gnb_params = ['model', 'name']
    
    model = request.args.get('model')
    name = request.args.get('name')
    models_params = [lr_params, rf_params, knn_params, gnb_params]
    mapping = { k:v for k,v in zip(models, models_params) }
    
    #Checking the existence of <model> and <name> parameter
    if model == None or name == None:
        
        result = {
                    'Result':'NOT OK',
                    'Data':"Please Insert <model> and <name> Paramaters"
                 }
        
        response = flask.jsonify(result)
        response.headers.set('Content-Type', 'application/json')
        
        return response
    
    #Checking the availability of the model
    if model.lower() not in models:
        
        result = {
                    'Result':'NOT OK',
                    'Data':'Model does not exist'
                 }
        
        response = flask.jsonify(result)
        response.headers.set('Content-Type', 'application/json')
        
        return response
    
    #Checking model parameters
    for param in request.args:
        if param not in mapping[model.lower()]:  
            
            result = {
                'Result':'NOT OK',
                'Data':'{}: Incorrect Parameter'.format(param)
             }
            
            response = flask.jsonify(result)
            response.headers.set('Content-Type', 'application/json')
            
            return response
    
    #Create the model if all checks passed
    result = {
                'Result':'OK',
                'Data':'{} model created'.format(model)
             }     
    
    response = flask.jsonify(result)
    response.headers.set('Content-Type', 'application/json')
            
    return response     



if __name__ == "__main__":
    app.run(host=ip_address, port=9999) 