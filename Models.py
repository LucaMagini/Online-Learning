import pickle
from river import optim
from river.preprocessing import StandardScaler
from river.multiclass import OneVsRestClassifier
from river.linear_model import LogisticRegression
from river.ensemble import AdaptiveRandomForestClassifier
from river.naive_bayes import GaussianNB
from river.neighbors import KNNClassifier

class Model:
    
    def __init__(self, model, name):
        
        #GENERAL PARAMETERS
        self.model = model
        self.name = name 
        #LOGISTIC REGRESSION PARAMETERS                      
        self.intercept_lr = 0.01                  
        self.loss = optim.losses.Log()  
        #RANDOM FOREST PARAMETERS                    
        self.n_models = 10                   
        self.split_criterion = 'info_gain' 
        #KNN PARAMETERS                 
        self.n_neighbors = 5
        self.p = 2
        
    def create_model(self):
        
        try:
            
            if self.model.lower() == 'logisticregression':
                #Constructing our pipeline (standardize features + model)
                model = (
                    StandardScaler() | 
                    OneVsRestClassifier(classifier=LogisticRegression(intercept_lr = self.intercept_lr, loss=self.loss))
                )
            
                with open('LogisticRegression_{}.pkl'.format(self.name), 'wb') as f:
                    pickle.dump(model, f)
                    
                return True    
                    
            elif self.model.lower() == 'randomforest':
                #Constructing our pipeline (standardize features + model)
                model = (
                    StandardScaler() | 
                    AdaptiveRandomForestClassifier(n_models = self.n_models, split_criterion = self.split_criterion, seed = 42)
                )
            
                with open('RandomForest_{}.pkl'.format(self.name), 'wb') as f:
                    pickle.dump(model, f)  
                    
                return True    
                    
            elif self.model.lower() == 'gaussiannb':
                #Constructing our pipeline (standardize features + model)
                model = (
                    StandardScaler() | 
                    GaussianNB()
                )
            
                with open('GaussianNB_{}.pkl'.format(self.name), 'wb') as f:
                    pickle.dump(model, f)
                    
                return True    
                    
            elif self.model.lower() == 'knn':
                #Constructing our pipeline (standardize features + model)
                model = (
                    StandardScaler() | 
                    KNNClassifier(n_neighbors = self.n_neighbors, p = self.p)
                )
            
                with open('KNN_{}.pkl'.format(self.name), 'wb') as f:
                    pickle.dump(model, f)
                    
                return True    
                    
            else:
                return False
        
        except Exception as e:
            
            print(e)
            return False
            
            
            