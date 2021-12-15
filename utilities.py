import glob

def check_list(x, elems):
    if x in elems or x == None:
        return True
    else:
        return False
    
def check_real(x):
    if isinstance(x, (float, int)) == True or x == None:
        return True
    return False
    
def check_natural(x):
    if (isinstance(x, int) == True and x >= 1) or x == None:
        return True
    return False

def check_natural_0(x):
    if (isinstance(x, int) == True and x >= 0) or x == None:
        return True
    return False

def cast_string(x, cast = 'float'):
    try:
        if cast == 'float':
            float(x)
            return float(x)
        elif cast == 'int':
            int(x)
            return int(x)
    except:
        return x

def check_models():

    lr_models = []
    rf_models = []
    gnb_models = []
    knn_models = []
    
    models_list = glob.glob("*/")
    for elem in models_list:
        end = elem.find("_")
        name = elem[:end]
        if name.lower() == 'logisticregression':
            start = elem.find("_")
            end = elem.find("'\'")
            name = elem[start+1:end]
            lr_models.append(name)
        elif name.lower() == 'randomforest':
            start = elem.find("_")
            end = elem.find("'\'")
            name = elem[start+1:end]
            rf_models.append(name)
        elif name.lower() == 'gaussiannb':
            start = elem.find("_")
            end = elem.find("'\'")
            name = elem[start+1:end]
            gnb_models.append(name)    
        elif name.lower() == 'knn':
            start = elem.find("_")
            end = elem.find("'\'")
            name = elem[start+1:end]
            knn_models.append(name)
    
    data = {}
    flag = False                     
            
    if lr_models != []:
        flag = True
        data['Logistic Regression Models'] = []
        for model_name in lr_models:
            data['Logistic Regression Models'].append(model_name)
    if rf_models != []:
        flag = True
        data['Random Forest Models'] = []
        for model_name in rf_models:
            data['Random Forest Models'].append(model_name)        
    if gnb_models != []:
        flag = True
        data['Gaussian Naive Bayes Models'] = []
        for model_name in gnb_models:
            data['Gaussian Naive Bayes Models'].append(model_name) 
    if knn_models != []:
        flag = True
        data['K-Nearest Neighbors Models'] = []
        for model_name in knn_models:
            data['K-Nearest Neighbors Models'].append(model_name)
            
    if flag == False:
        data = 'THERE IS NO MODEL'
        
    result = {
                'Result':'OK',
                'Data': data
             }   
    
    return result