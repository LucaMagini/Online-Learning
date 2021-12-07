import pickle, json, sys #sys.stdout.flush() [To print on console]
from utilities import check_list, check_real, check_natural, check_natural_0
from river import optim
from river.preprocessing import StandardScaler
from river.multiclass import OneVsRestClassifier
from river.linear_model import LogisticRegression
from river.ensemble import AdaptiveRandomForestClassifier
from river.naive_bayes import GaussianNB
from river.neighbors import KNNClassifier

with open('config.json') as json_file:
    data = json.load(json_file)
    
default = {}
for k in data:
    for key in data[k]:
        default[key] = None if data[k][key]['default'] == 'None' else data[k][key]['default']     

class Model:
    
    def __init__(self):
        
        #GENERAL PARAMETERS
        self.model_type = None
        self.name = None
        self.model = None
        self.data = data
      
    def create_logisticregression(self, model_type, name,
                                  optimizer=default['optimizer'],
                                  loss=default['loss'],
                                  l2=default['l2'], 
                                  intercept_init=default['intercept_init'],
                                  intercept_lr=default['intercept_lr'],
                                  clip_gradient=default['clip_gradient'],
                                  initializer=default['initializer']
                                  ):
        
        for k in data['LogisticRegression']:
            result = {
                'Result':'NOT OK',
                'Data':'{}: Incorrect Parameter Value'.format(k)
             }
            if data['LogisticRegression'][k]['values'] == ['R']:
                res = check_real(eval(k))
                if res == False:
                    return False, result
            elif data['LogisticRegression'][k]['values'] == ['N']:
                res = check_natural(eval(k))
                if res == False:
                    return False, result
            else:
                res = check_list(eval(k), data['LogisticRegression'][k]['values'])
                if res == False:
                    return False, result
        
        
        model = StandardScaler() | OneVsRestClassifier(
                classifier=LogisticRegression(
                    optimizer = optimizer,
                    loss = loss,
                    l2 = l2,
                    intercept_init = intercept_init,
                    intercept_lr = intercept_lr,
                    clip_gradient = clip_gradient,
                    initializer = initializer
                ))
        
        self.model_type = model_type
        self.name = name
        self.model = model
        
        with open('{}_{}.pkl'.format(self.model_type, self.name), 'wb') as f:
                pickle.dump(self.model, f)
                
        return True, self.model
    
    def create_randomforest(self, model_type, name,
                                  n_models=default['n_models'],
                                  max_features=default['max_features'],
                                  lambda_value=default['lambda_value'], 
                                  metric=default['metric'],
                                  disable_weighted_vote=default['disable_weighted_vote'],
                                  grace_period=default['grace_period'],
                                  max_depth=default['max_depth'],
                                  split_criterion=default['split_criterion'],
                                  leaf_prediction=default['leaf_prediction'],
                                  nb_threshold=default['nb_threshold'],
                                  max_size=default['max_size'],
                                  seed=default['seed']
                                  ):
        
        for k in data['RandomForest']:
            result = {
                'Result':'NOT OK',
                'Data':'{}: Incorrect Parameter Value'.format(k)
             }
            if data['RandomForest'][k]['values'] == ['R']:
                res = check_real(eval(k))
                if res == False:
                    return False, result
            elif data['RandomForest'][k]['values'] == ['N']:
                res = check_natural(eval(k))
                if res == False:
                    return False, result
            elif data['RandomForest'][k]['values'] == ['N0']:
                res = check_natural_0(eval(k))
                if res == False:
                    return False, result
            elif data['RandomForest'][k]['values'] == ['N_or_list']:
                res1 = check_natural(eval(k))
                res2 = check_list(eval(k), data['RandomForest'][k]['values'][1:])
                if res1 == False and res2 == False:
                    return False, result    
            else:
                res = check_list(eval(k), data['RandomForest'][k]['values'])
                if res == False:
                    return False, result
        
        
        model = StandardScaler() | AdaptiveRandomForestClassifier(
            n_models = n_models,
            max_features = max_features,
            lambda_value = lambda_value, 
            metric = metric,
            disable_weighted_vote = disable_weighted_vote,
            grace_period = grace_period,
            max_depth = max_depth,
            split_criterion = split_criterion,
            leaf_prediction = leaf_prediction,
            nb_threshold = nb_threshold,
            max_size = max_size,
            seed = seed
            )
        
        self.model_type = model_type
        self.name = name
        self.model = model
        
        with open('{}_{}.pkl'.format(self.model_type, self.name), 'wb') as f:
                pickle.dump(self.model, f)
                
        return True, self.model
    
    def create_gaussiannb(self, model_type, name):
        
        model = StandardScaler() | GaussianNB()
        
        self.model_type = model_type
        self.name = name
        self.model = model
        
        with open('{}_{}.pkl'.format(self.model_type, self.name), 'wb') as f:
                pickle.dump(self.model, f)
                
        return True, self.model
    
    
    def create_knn(self, model_type, name,
                         n_neighbors=default['n_neighbors'],
                         window_size=default['window_size'],
                         leaf_size=default['leaf_size'], 
                         p=default['p']
                        ):
        
        for k in data['KNN']:
            result = {
                'Result':'NOT OK',
                'Data':'{}: Incorrect Parameter Value'.format(k)
             }
            if data['KNN'][k]['values'] == ['R']:
                res = check_real(eval(k))
                if res == False:
                    return False, result
            elif data['KNN'][k]['values'] == ['N']:
                res = check_natural(eval(k))
                if res == False:
                    return False, result  
            else:
                res = check_list(eval(k), data['RandomForest'][k]['values'])
                if res == False:
                    return False, result
        
        
        model = StandardScaler() | KNNClassifier(
            n_neighbors = n_neighbors,
            window_size = window_size,
            leaf_size = leaf_size, 
            p = p
            )
        
        self.model_type = model_type
        self.name = name
        self.model = model
        
        with open('{}_{}.pkl'.format(self.model_type, self.name), 'wb') as f:
                pickle.dump(self.model, f)
                
        return True, self.model
        