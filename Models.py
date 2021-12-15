import pickle, json, os, sys #sys.stdout.flush() [To print on console]
from utilities import check_list, check_real, check_natural, check_natural_0
from river import optim
from river.preprocessing import StandardScaler
from river.multiclass import OneVsRestClassifier
from river.linear_model import LogisticRegression
from river.ensemble import AdaptiveRandomForestClassifier
from river.naive_bayes import GaussianNB
from river.neighbors import KNNClassifier
from river.metrics import Accuracy
from datetime import datetime

with open('config.json') as json_file:
    data = json.load(json_file)
with open('dataset_info.json') as json_file:
    dataset_info = json.load(json_file)    
    
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
        self.metric = Accuracy()
        self.data = data
        self.dataset_info = dataset_info
        self.attempts_info = {
            'attempts': 0,
            'ok_predictions': 0,
            'created_on': str(datetime.now())[:-7],
            'last_edit': str(datetime.now())[:-7]
            }
      
    def create_logisticregression(self, model_type, name,
                                  optimizer=default['optimizer'],
                                  loss=default['loss'],
                                  l2=default['l2'], 
                                  intercept_init=default['intercept_init'],
                                  intercept_lr=default['intercept_lr'],
                                  clip_gradient=default['clip_gradient'],
                                  initializer=default['initializer']
                                  ):
        
        self.reset_model()
        
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
                    optimizer = eval('optim.' + optimizer +'()') if optimizer != None else None,
                    loss = eval('optim.losses.' + loss + '()') if loss != None else None,
                    l2 = l2,
                    intercept_init = intercept_init,
                    intercept_lr = intercept_lr,
                    clip_gradient = clip_gradient,
                    initializer = eval('optim.initializers.' + initializer + '()') if initializer != None else None,
                ))
        
        self.model_type = model_type
        self.name = name
        self.model = model
        
        filename = "./LOGISTICREGRESSION_{}".format(self.name)
        os.makedirs(filename, exist_ok=True)
        
        with open('./LOGISTICREGRESSION_{}/model_{}.pkl'.format(self.name, self.name), 'wb') as f:
                pickle.dump(self.model, f)
        with open('./LOGISTICREGRESSION_{}/metric_{}.pkl'.format(self.name, self.name), 'wb') as m:
                pickle.dump(self.metric, m)
        with open('./LOGISTICREGRESSION_{}/info_{}.json'.format(self.name, self.name), 'w') as i:
                json.dump(self.attempts_info, i)        
                
        return True, self.model
    
    def create_randomforest(self, model_type, name,
                                  n_models=default['n_models'],
                                  max_features=default['max_features'],
                                  lambda_value=default['lambda_value'],                                
                                  disable_weighted_vote=default['disable_weighted_vote'],
                                  grace_period=default['grace_period'],
                                  max_depth=default['max_depth'],
                                  split_criterion=default['split_criterion'],
                                  leaf_prediction=default['leaf_prediction'],
                                  nb_threshold=default['nb_threshold'],
                                  max_size=default['max_size'],
                                  seed=default['seed']
                                  ):
        
        self.reset_model()
        
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
        
        filename = "./RANDOMFOREST_{}".format(self.name)
        os.makedirs(filename, exist_ok=True)
        
        with open('./RANDOMFOREST_{}/model_{}.pkl'.format(self.name, self.name), 'wb') as f:
                pickle.dump(self.model, f)
        with open('./RANDOMFOREST_{}/metric_{}.pkl'.format(self.name, self.name), 'wb') as m:
                pickle.dump(self.metric, m)
        with open('./RANDOMFOREST_{}/info_{}.json'.format(self.name, self.name), 'w') as i:
                json.dump(self.attempts_info, i)               
                
        return True, self.model
    
    def create_gaussiannb(self, model_type, name):
        
        self.reset_model()
        
        model = StandardScaler() | GaussianNB()
        
        self.model_type = model_type
        self.name = name
        self.model = model
        
        filename = "./GAUSSIANNB_{}".format(self.name)
        os.makedirs(filename, exist_ok=True)
        
        with open('./GAUSSIANNB_{}/model_{}.pkl'.format(self.name, self.name), 'wb') as f:
                pickle.dump(self.model, f)
        with open('./GAUSSIANNB_{}/metric_{}.pkl'.format(self.name, self.name), 'wb') as m:
                pickle.dump(self.metric, m)
        with open('./GAUSSIANNB_{}/info_{}.json'.format(self.name, self.name), 'w') as i:
                json.dump(self.attempts_info, i)               
                
        return True, self.model
    
    
    def create_knn(self, model_type, name,
                         n_neighbors=default['n_neighbors'],
                         window_size=default['window_size'],
                         leaf_size=default['leaf_size'], 
                         p=default['p']
                        ):
        
        self.reset_model()
        
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
        
        filename = "./KNN_{}".format(self.name)
        os.makedirs(filename, exist_ok=True)
        
        with open('./KNN_{}/model_{}.pkl'.format(self.name, self.name), 'wb') as f:
                pickle.dump(self.model, f)
        with open('./KNN_{}/metric_{}.pkl'.format(self.name, self.name), 'wb') as m:
                pickle.dump(self.metric, m)
        with open('./KNN_{}/info_{}.json'.format(self.name, self.name), 'w') as i:
                json.dump(self.attempts_info, i)               
                
        return True, self.model
    
    
    def reset_model(self):
        
        self.model_type = None
        self.name = None
        self.model = None       
        self.metric = Accuracy()
        self.data = data
        self.dataset_info = dataset_info
        self.attempts_info = {
            'attempts': 0,
            'ok_predictions': 0,
            'created_on': str(datetime.now())[:-7],
            'last_edit': str(datetime.now())[:-7]
            }
    
    
    def load_model(self, model_type, name):
        
        model_type = model_type.upper()
        
        try:
            
            with open('./{}_{}/model_{}.pkl'.format(model_type, name, name), 'rb') as f:
                self.model = pickle.load(f)
            with open('./{}_{}/metric_{}.pkl'.format(model_type, name, name), 'rb') as m:
                self.metric = pickle.load(m)
            with open('./{}_{}/info_{}.json'.format(model_type, name, name), 'r') as i:
                self.attempts_info = json.load(i)           
                
            self.model_type = model_type
            self.name = name
            
            result = {
                'Result': 'OK',
                'Data': '{} model correctly loaded'.format(self.name) 
             }
            
        except:
            
            result = {
                'Result': 'NOT OK',
                'Data': '{} model does not exists'.format(name) 
             }
            
        finally:
            
            return result
        
        
    def model_info(self, model_type, name):
        
        self.load_model(model_type, name)
        
        result = {
                'Result':'OK',
                'Data':{
                    'Name': self.name,
                    'Type': self.model_type,
                    'Accuracy': str(self.metric)[10:] + ' -- ({}/{})'.format(self.attempts_info['ok_predictions'],
                                                                             self.attempts_info['attempts']),
                    'Created On': self.attempts_info['created_on'],
                    'Last Edit' : self.attempts_info['last_edit']
                    }
                }
        
        return result
            
        
    def inference(self, observation):
        
        observation = [ float(item) for item in observation ]
        
        entity = { k:v for k,v in zip(self.dataset_info['features'][:-1], observation[:-1]) }
        target = int(observation[-1])
        
        
        print(self.model['OneVsRestClassifier'].classifier.loss)
        print(type(self.model['OneVsRestClassifier'].classifier.loss))
        
        preds = self.model.predict_one(entity)
        self.model = self.model.learn_one(entity, target)
        self.metric = self.metric.update(target, preds)
        
        self.attempts_info['attempts'] += 1
        if preds == int(observation[-1]):
            self.attempts_info['ok_predictions'] += 1
        self.attempts_info['last_edit'] = str(datetime.now())[:-7]   
            
        with open('./{}_{}/model_{}.pkl'.format(self.model_type.upper(), self.name, self.name), 'wb') as f:
                pickle.dump(self.model, f)
        with open('./{}_{}/metric_{}.pkl'.format(self.model_type.upper(), self.name, self.name), 'wb') as m:
                pickle.dump(self.metric, m)
        with open('./{}_{}/info_{}.json'.format(self.model_type.upper(), self.name, self.name), 'w') as i:
                json.dump(self.attempts_info, i)               
        
        result = {
                'Result':'OK',
                'Data':{
                    'Target Value': target,
                    'Prediction': preds,
                    'Accuracy': str(self.metric)[10:] + ' -- ({}/{})'.format(self.attempts_info['ok_predictions'],
                                                                             self.attempts_info['attempts']) 
                    }
                }
        
        return result
                
                
    
    

            
            
            
        
        
        
        
    
    
    
    
    
    
