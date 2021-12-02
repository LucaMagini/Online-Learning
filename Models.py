import pickle
from river import optim
from river.preprocessing import StandardScaler
from river.multiclass import OneVsRestClassifier
from river.linear_model import LogisticRegression
from river.ensemble import AdaptiveRandomForestClassifier
from river.naive_bayes import GaussianNB
from river.neighbors import KNNClassifier

class Model:
    
    def __init__(self):
        
        #GENERAL PARAMETERS
        self.model = None
        self.name = None
        self.list = ['logisticregression', 'randomforest', 
                       'knn', 'gaussiannb']
        #LOGISTIC REGRESSION PARAMETERS                      
        self.intercept_lr = None                  
        self.loss = None  
        #RANDOM FOREST PARAMETERS                    
        self.n_models = None                   
        self.split_criterion = None 
        #KNN PARAMETERS                 
        self.n_neighbors = None
        self.p = None
        
    def create_model( self, model, name, intercept_lr, 
                      loss, n_models,
                      split_criterion, n_neighbors,
                      p ):
        
        self.model = model
        self.name = name
        self.intercept_lr = intercept_lr
        self.n_models = n_models
        self.split_criterion = split_criterion
        self.n_neighbors = n_neighbors
        self.p = p
        if loss == 'Hinge':
            self.loss = optim.losses.Hinge()
        else:
            self.loss = optim.losses.Log()
        
        classes = [ OneVsRestClassifier(classifier=LogisticRegression(intercept_lr = self.intercept_lr, loss=self.loss)), 
                    AdaptiveRandomForestClassifier(n_models = self.n_models, split_criterion = self.split_criterion, seed = 42), 
                    GaussianNB(),
                    KNNClassifier(n_neighbors = self.n_neighbors, p = self.p) ]
        
        mapping = { k:v for k,v in zip(self.list, classes) }
        
        if self.model.lower() in mapping:
            #Constructing our pipeline (standardize features + model)
            _model = (
                StandardScaler() | 
                mapping[self.model.lower()]
            )
        
            with open('{}_{}.pkl'.format(self.model, self.name), 'wb') as f:
                pickle.dump(_model, f)
                
            return True, _model
        
        else:
            
            return False
            
        # if self.model.lower() == 'logisticregression':
        #     #Constructing our pipeline (standardize features + model)
        #     model = (
        #         StandardScaler() | 
        #         OneVsRestClassifier(classifier=LogisticRegression(intercept_lr = self.intercept_lr, loss=self.loss))
        #     )
        
        #     with open('LogisticRegression_{}.pkl'.format(self.name), 'wb') as f:
        #         pickle.dump(model, f)
                
        #     return True    
                
        # elif self.model.lower() == 'randomforest':
        #     #Constructing our pipeline (standardize features + model)
        #     model = (
        #         StandardScaler() | 
        #         AdaptiveRandomForestClassifier(n_models = self.n_models, split_criterion = self.split_criterion, seed = 42)
        #     )
        
        #     with open('RandomForest_{}.pkl'.format(self.name), 'wb') as f:
        #         pickle.dump(model, f)  
                
        #     return True    
                
        # elif self.model.lower() == 'gaussiannb':
        #     #Constructing our pipeline (standardize features + model)
        #     model = (
        #         StandardScaler() | 
        #         GaussianNB()
        #     )
        
        #     with open('GaussianNB_{}.pkl'.format(self.name), 'wb') as f:
        #         pickle.dump(model, f)
                
        #     return True    
                
        # elif self.model.lower() == 'knn':
        #     #Constructing our pipeline (standardize features + model)
        #     model = (
        #         StandardScaler() | 
        #         KNNClassifier(n_neighbors = self.n_neighbors, p = self.p)
        #     )
        
        #     with open('KNN_{}.pkl'.format(self.name), 'wb') as f:
        #         pickle.dump(model, f)
                
        #     return True    
        
            
            
            