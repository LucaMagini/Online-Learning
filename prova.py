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

dataset = None

@app.route('/api/v1/try', methods=['GET'])
@auto.doc()
def x():
    
    global dataset
    dataset = '5'
    
    res = {'Data': dataset }
              
    response = flask.jsonify(res)
    response.headers.set('Content-Type', 'application/json')    

    return response

@app.route('/api/v1/try2', methods=['GET'])
@auto.doc()
def y():
    
    res = {'Data': dataset }
              
    response = flask.jsonify(res)
    response.headers.set('Content-Type', 'application/json')    

    return response

if __name__ == "__main__":
    app.run(host=ip_address, port=3377) 