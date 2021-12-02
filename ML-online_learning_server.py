import flask
from flask_selfdoc import Autodoc
from flask import request
from Models import Model

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False
auto = Autodoc(app)

ip_address = '0.0.0.0'

ML_model = Model()

@app.route('/', methods = ['GET'])
@auto.doc()
def home():
    "Checking Server Operation"
    
    try:
        result = {
                    'Result':'OK',
                    'Data':'Server Ready'
                 }
            
    except:
        result = {
                    'Result':'NOT OK',
                    'Data':'Server Not Ready'
                 }
            
    response = flask.jsonify(result)
    response.headers.set('Content-Type', 'application/json')
    
    return response  

@app.route('/api/v1/create_model', methods = ['GET'])
@auto.doc(args=['model', 'name', 'intercept_lr', 'loss', 'n_models', 
                'split_criterion', 'n_neighbors', 'p'])
def create_model():
    "Create a ML Model for Online Learning"
    
    models = ML_model.list
    
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
                    'Data':'Please Insert <model> and <name> Paramaters'
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
    model = request.args.get('model')
    name = request.args.get('name')
    intercept_lr = request.args.get('intercept_lr') if request.args.get('intercept_lr') != None else 0.01    
    loss = request.args.get('loss') if request.args.get('loss') != None else 'Log'
    n_models = request.args.get('n_models') if request.args.get('n_models') != None else 10
    split_criterion = request.args.get('split_criterion') if request.args.get('split_criterion') != None else 'info_gain'
    n_neighbors = request.args.get('n_neighbors') if request.args.get('n_neighbors') != None else 5
    p = request.args.get('p') if request.args.get('p') != None else 2
    
    try:
        
        res, _model = ML_model.create_model(model, name, intercept_lr, loss,
                                    n_models, split_criterion,
                                    n_neighbors, p)
        
        if res == True:
            
            if model.lower() == 'logisticregression':
                
                if request.args.get('intercept_lr') != None:
                    _intercept = _model['OneVsRestClassifier'].classifier.intercept_lr
                else:
                    _intercept = str(_model['OneVsRestClassifier'].classifier.intercept_lr.learning_rate)
                
                data = {'Model': model,
                        'Name': name,
                        'Parameters': { 'intercept_lr': _intercept,
                                        'loss': str(_model['OneVsRestClassifier'].classifier.loss)
                            }
                        }
                
            elif model.lower() == 'randomforest':
                
                data = {'Model': model,
                        'Name': name,
                        'Parameters': { 'n_models': _model['AdaptiveRandomForestClassifier'].n_models,
                                        'split_criterion': _model['AdaptiveRandomForestClassifier'].split_criterion
                            }
                        }
                
            elif model.lower() == 'gaussiannb':
                
                data = {'Model': model,
                        'Name': name,
                        }  
                
            elif model.lower() == 'knn':
                
                data = {'Model': model,
                        'Name': name,
                        'Parameters': { 'n_neighbors': _model['KNNClassifier'].n_neighbors,
                                        'p': _model['KNNClassifier'].p
                            }
                        }     
            
                
            result = {
                        'Result':'OK',
                        'Data': data
                     }  
            
            response = flask.jsonify(result)
            response.headers.set('Content-Type', 'application/json')
            
            return response
        
    except Exception as e:
        
        print(e)
        result = {
                        'Result':'NOT OK',
                        'Data':'Model not created'
                  }  
            
        response = flask.jsonify(result)
        response.headers.set('Content-Type', 'application/json')
        
        return response    



if __name__ == "__main__":
    app.run(host=ip_address, port=9999) 