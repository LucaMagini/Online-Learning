import requests, json, argparse, threading
import pandas as pd
from Models import Model

parser = argparse.ArgumentParser(description='''Simulate data stream for online learning method and 
        get the prediction and total accuracy of the model''')
parser.add_argument('model_type', help='[LogisticRegression, RandomForest, KNN, GaussianNB]')
parser.add_argument('name', help='The name of the model')
parser.add_argument('start', type=int, help='Dataset starting line for the flow')
parser.add_argument('interval', type=float, help='Time interval of the flow in seconds')
args = parser.parse_args()

model = Model()

with open('dataset_info.json') as json_file:
        data = json.load(json_file)
with open('config_model.json') as params_file:
        model_param = json.load(params_file)        
        
dataset = pd.read_csv(data['dataset'])
start = 0

def setup():
        
    model_type = args.model_type
    name = args.name
    df_start = args.start
    interval = args.interval
    
    global dataset
    dataset = dataset[df_start:]
    
    result = requests.get("http://localhost:9999/api/v1/load_model?model_type={}&name={}".format(model_type, name))
    result = result.json()
    
    message = result['Data']
    result['Data'] = {'Message': message,
                      'Dataset': data['dataset'],
                      'Starting_Row': start,
                      'Interval': interval
                      }
    
    print(json.dumps(
        result,
        sort_keys=False,
        indent=4,
        separators=(',', ': ')
    ))
    print('------------------------------')
    
    if result['Result'] == 'NOT OK':
        
        url = "http://localhost:9999/api/v1/create_model?model_type={}&name={}".format(model_type, name) 
        for param in model_param[model_type]:
            if model_param[model_type][param] != "None":
                elem = '&{}={}'.format(param, model_param[model_type][param])
                url += elem
        
        result = requests.get(url)
        result = result.json()
        
        print(json.dumps(
            result,
            sort_keys=False,
            indent=4,
            separators=(',', ': ')
        ))
        print('------------------------------')


def data_stream(n_row):
    observ = dataset.iloc[n_row].tolist()
    data = {"values": observ}
    
    result = requests.post("http://localhost:9999/api/v1/inference", json = data)
    result = result.json()
    
    print(json.dumps(
        result,
        sort_keys=False,
        indent=4,
        separators=(',', ': ')
    ))
    print('------------------------------')
    
    global start
    start += 1
    
    threading.Timer(args.interval, data_stream, [start]).start()

#SIMULATING DATA FLOW 
   
value = setup()
data_stream(start)

