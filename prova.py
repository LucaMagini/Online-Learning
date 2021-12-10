import json, glob
from Models import *


with open('config.json') as json_file:
    data = json.load(json_file)  
        
    
ret = glob.glob('*.pkl')
print(ret)

lr = []
knn = []

for elem in knn:
    print('ok')
# for elem in ret:
#     end = elem.find("_")
#     name = elem[:end]
#     if name.lower() == 'logisticregression':
#         start = elem.find("_")
#         end = elem.find(".")
#         name = elem[start+1:end]
#         lr.append(name)
#     elif name.lower() == 'knn':
#         start = elem.find("_")
#         end = elem.find(".")
#         name = elem[start+1:end]
#         knn.append(name)    
        
# print(lr) 
# print(knn)       