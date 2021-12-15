import json
from Models import *


with open('config.json') as json_file:
    data = json.load(json_file)  
        
from datetime import datetime

today = str(datetime.now())[:-7]
print(today)