'''
create item graph for each item through their features
'''

import json
import numpy as np
from tqdm import tqdm

ITEM_GRAPH_PATH = "./itemGraph/"
with open("./items.json",'r') as load_f:
    items_info = json.load(load_f)

for key in items_info.keys():
    items_info[key].pop('like')
    items_info[key].pop('dislike')



def createItemGraph(uid,item_info,info_type):
    if len(item_info) == 0:
        return 
    
    save_json_path = ITEM_GRAPH_PATH + info_type + "/" + str(uid) + ".json"
    _edges = []
    _features = {}
    
    _features["0"] = str(uid)
    for item_id in item_info:
        _edges.append([0,int(item_id)])
        _features[str(item_id)] = str(item_id)
    
#     print(_edges,_features)
    json_content = json.dumps(dict(edges = _edges,features = _features))
    with open(save_json_path, 'w') as f:
        f.write(json_content)

for item_id in tqdm(items_info.keys()):
    id_item = items_info.get(str(item_id),{})
#     print(id_user)
#    like_info     =  np.array(id_item['like'])
#    dislike_info  =  np.array(id_item['dislike'])
    finish_info   =  np.array(id_item['finish'])
    unfinish_info =  np.array(id_item['unfinish'])
#     print(len(like_info),len(dislike_info),len(finish_info),len(unfinish_info))
    
#    createItemGraph(item_id,like_info,"like")
#    createItemGraph(item_id,dislike_info,"dislike")

    if len(finish_info) >= 5:
        createItemGraph(item_id,finish_info,"finish")
    if len(unfinish_info) >= 5:
        createItemGraph(item_id,unfinish_info,"unfinish")


