import flask, json
import pandas as pd
from flask_selfdoc import Autodoc
from flask import request
from Models import Model

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False
auto = Autodoc(app)

ip_address = '0.0.0.0'

model = Model()

with open('dataset_info.json') as json_file:
    data = json.load(json_file)

dataset = pd.read_csv(data['dataset'])

@app.route('/api/v1/model_selection', methods = ['GET'])
@auto.doc(args=['model_type', 'name', 'dataset', 'start'])
                                  
def model_selection():
    "Shows the list of all the existing models" 
    

    
    response = flask.jsonify(result)
    response.headers.set('Content-Type', 'application/json')
    
    return response



@app.route('/api/v1/data_stream', methods=['POST'])
@auto.doc()
def data_stream():
    """Simulate data stream for online learning method and 
       get the prediction and total accuracy of the model"""
    
    req_data = request.get_json()
    res = model.inference(req_data['values'])
               
    
    response = flask.jsonify(res)
    response.headers.set('Content-Type', 'application/json')    

    return response















if __name__ == "__main__":
    app.run(host=ip_address, port=3333) 