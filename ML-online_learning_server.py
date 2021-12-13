import flask, sys #sys.stdout.flush() [To print on console]
from flask_selfdoc import Autodoc
from flask import request
from Models import Model, default
from utilities import cast_string, check_models

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False
auto = Autodoc(app)

ip_address = '0.0.0.0'

model = Model()

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
@auto.doc(args=['model_type', 'name', 'optimizer', 'loss', 'l2',
                'intercept_init', 'intercept_lr', 'clip_gradient',
                'initializer'])
                                  
def create_model():
    "Create a ML Model for Online Learning"
    
    models = [model.lower() for model in model.data]
    
    lr_params = [ param for param in model.data['LogisticRegression'] ]
    rf_params = [ param for param in model.data['RandomForest'] ]
    knn_params = [ param for param in model.data['KNN'] ]
    gnb_params = []
    
    model_type = request.args.get('model_type')
    name = request.args.get('name')
    models_params = [ lr_params, rf_params, knn_params, gnb_params ]
    mapping = { k:v for k,v in zip(models, models_params) }
    
    #Checking the existence of <model> and <name> parameter
    if model_type == None or name == None:
        
        result = {
                    'Result':'NOT OK',
                    'Data':'Please Insert <model_type> and <name> Paramaters'
                 }
        
        response = flask.jsonify(result)
        response.headers.set('Content-Type', 'application/json')
        
        return response
    
    #Checking the availability of the model
    if model_type.lower() not in models:
        
        result = {
                    'Result':'NOT OK',
                    'Data':'Model does not exist'
                 }
        
        response = flask.jsonify(result)
        response.headers.set('Content-Type', 'application/json')
        
        return response
    
    #Checking model parameters
    for param in request.args:
        if param not in mapping[model_type.lower()] and param not in ['model_type', 'name']:  
            
            result = {
                'Result':'NOT OK',
                'Data':'{}: Incorrect Parameter'.format(param)
             }
            
            response = flask.jsonify(result)
            response.headers.set('Content-Type', 'application/json')
            
            return response
    
    #Create the model if all checks passed
    
    #LogisticRegression Parameters
    optimizer = request.args.get('optimizer') if request.args.get('optimizer') != None else default['optimizer']
    loss = request.args.get('loss') if request.args.get('loss') != None else default['loss']
    l2 = cast_string(request.args.get('l2')) if request.args.get('l2') != None else float(default['l2'])
    intercept_init = cast_string(request.args.get('intercept_init')) if request.args.get('intercept_init') != None else float(default['intercept_init'])   
    intercept_lr = cast_string(request.args.get('intercept_lr')) if request.args.get('intercept_lr') != None else float(default['intercept_lr'])    
    clip_gradient = cast_string(request.args.get('clip_gradient')) if request.args.get('clip_gradient') != None else float(default['clip_gradient'])    
    initializer = request.args.get('initializer') if request.args.get('initializer') != None else default['initializer']
    
    #RandomForest Parameters
    n_models = cast_string(request.args.get('n_models'), cast='int') if request.args.get('n_models') != None else int(default['n_models'])
    max_features = request.args.get('max_features') if request.args.get('max_features') != None else default['max_features']
    lambda_value = cast_string(request.args.get('lambda_value'), cast='int') if request.args.get('lambda_value') != None else int(default['lambda_value']) 
    metric = request.args.get('metric') if request.args.get('metric') != None else default['metric']
    disable_weighted_vote = request.args.get('disable_weighted_vote') if request.args.get('disable_weighted_vote') != None else default['disable_weighted_vote']
    grace_period = cast_string(request.args.get('grace_period'), cast='int') if request.args.get('grace_period') != None else int(default['grace_period'])
    max_depth = cast_string(request.args.get('max_depth'), cast='int') if request.args.get('max_depth') != None else default['max_depth']
    split_criterion = request.args.get('split_criterion') if request.args.get('split_criterion') != None else default['split_criterion']
    leaf_prediction = request.args.get('leaf_prediction') if request.args.get('leaf_prediction') != None else default['leaf_prediction']
    nb_threshold = cast_string(request.args.get('nb_threshold'), cast='int') if request.args.get('nb_threshold') != None else int(default['nb_threshold'])
    max_size = cast_string(request.args.get('max_size'), cast='int') if request.args.get('max_size') != None else int(default['max_size'])
    seed = cast_string(request.args.get('seed'), cast='int') if request.args.get('seed') != None else default['seed']
    
    
    #KNN Parameters
    n_neighbors = cast_string(request.args.get('n_neighbors'), cast='int') if request.args.get('n_neighbors') != None else int(default['n_neighbors'])
    window_size = cast_string(request.args.get('window_size'), cast='int') if request.args.get('window_size') != None else int(default['window_size'])
    leaf_size = cast_string(request.args.get('leaf_size'), cast='int') if request.args.get('leaf_size') != None else int(default['leaf_size'])
    p = cast_string(request.args.get('p'), cast='int') if request.args.get('p') != None else int(default['p'])
    
    
    params_logisticregression = [ model_type, name, optimizer, loss, l2, intercept_init,
                                  intercept_lr, clip_gradient, initializer ]
    params_randomforest = [ model_type, name, n_models, max_features, lambda_value,
                                metric, disable_weighted_vote, grace_period, max_depth,
                                split_criterion, leaf_prediction, nb_threshold,
                                max_size, seed ]
    params_knn = [ model_type, name, n_neighbors, window_size, leaf_size, p ]
    params_gaussiannb = [ model_type, name ]
    
    #try:      
    func_name = eval("model.create_" + model_type.lower())
    params_name = eval("params_" + model_type.lower())
    
    res = func_name(*params_name)
    
    if res[0] == True:
        
        _model = res[1]
        
        if model_type.lower() == 'logisticregression':
            
            # if request.args.get('intercept_lr') != None:
            #     _intercept = _model['OneVsRestClassifier'].classifier.intercept_lr
            # else:
            #     _intercept = str(_model['OneVsRestClassifier'].classifier.intercept_lr.learning_rate)
            
            info = {'Model': model_type,
                    'Name': name,
                    'Parameters': { 
                        'optimizer':str(_model['OneVsRestClassifier'].classifier.optimizer),
                        'loss': str(_model['OneVsRestClassifier'].classifier.loss),
                        'l2': _model['OneVsRestClassifier'].classifier.l2,
                        'intercept_init':_model['OneVsRestClassifier'].classifier.intercept_init,
                        #'intercept_lr': _intercept,
                        'intercept_lr': str(_model['OneVsRestClassifier'].classifier.intercept_lr.learning_rate),
                        'clip_gradient': _model['OneVsRestClassifier'].classifier.clip_gradient,
                        'initializer':str(_model['OneVsRestClassifier'].classifier.initializer)
                        }
                    }
            
        elif model_type.lower() == 'randomforest':
            
            info = {'Model': model_type,
                    'Name': name,
                    'Parameters': { 
                        'n_models': _model['AdaptiveRandomForestClassifier'].n_models,
                        'max_features': _model['AdaptiveRandomForestClassifier'].max_features,
                        'lambda_value': _model['AdaptiveRandomForestClassifier'].lambda_value, 
                        'metric': str(_model['AdaptiveRandomForestClassifier'].metric),
                        'disable_weighted_vote': _model['AdaptiveRandomForestClassifier'].disable_weighted_vote,
                        'grace_period': _model['AdaptiveRandomForestClassifier'].grace_period,
                        'max_depth': _model['AdaptiveRandomForestClassifier'].max_depth,
                        'split_criterion': _model['AdaptiveRandomForestClassifier'].split_criterion,
                        'leaf_prediction': _model['AdaptiveRandomForestClassifier'].leaf_prediction,
                        'nb_threshold': _model['AdaptiveRandomForestClassifier'].nb_threshold,
                        'max_size': _model['AdaptiveRandomForestClassifier'].max_size,
                        'seed': _model['AdaptiveRandomForestClassifier'].seed
                        }
                    }
            
        elif model_type.lower() == 'gaussiannb':
            
            info = {'Model': model_type,
                    'Name': name,
                    'Parameters': 'There are no parameters available for this model'
                    }  
            
        elif model_type.lower() == 'knn':
            
            info = {'Model': model_type,
                    'Name': name,
                    'Parameters': { 
                        'n_neighbors': _model['KNNClassifier'].n_neighbors,
                        'window_size': _model['KNNClassifier'].window_size,
                        'leaf_size': _model['KNNClassifier'].leaf_size,
                        'p': _model['KNNClassifier'].p
                        }
                    }     
        
        print(info)
        sys.stdout.flush()
             
        result = {
                    'Result':'OK',
                    'Data': info
                 }  
        
        response = flask.jsonify(result)
        response.headers.set('Content-Type', 'application/json')
        
        return response
    
    else:
        return res[1]
            
    # except Exception as e:
        
    #     print(e)
    #     result = {
    #                     'Result':'NOT OK',
    #                     'Data':'Model not created'
    #               }  
            
    #     response = flask.jsonify(result)
    #     response.headers.set('Content-Type', 'application/json')
        
    #     return response 
    
@app.route('/api/v1/existing_models', methods = ['GET'])
@auto.doc()
                                  
def existing_models():
    "Shows the list of all the existing models" 
    
    result = check_models()
    
    response = flask.jsonify(result)
    response.headers.set('Content-Type', 'application/json')
    
    return response

@app.route('/api/v1/load_model', methods = ['GET'])
@auto.doc(args=['model_type', 'name'])
                                  
def loading_model():
    "Load the model" 
    
    model_type = request.args.get('model_type')
    name = request.args.get('name')
    
    if model_type == None or name == None:
        
        result = {
                'Result': 'NOT OK',
                'Data': 'Please insert <model_type> and <name> parameters' 
             }
    else:    
        result = model.load_model(model_type, name)
        
    response = flask.jsonify(result)
    response.headers.set('Content-Type', 'application/json')
    
    return response


if __name__ == "__main__":
    app.run(host=ip_address, port=9999) 