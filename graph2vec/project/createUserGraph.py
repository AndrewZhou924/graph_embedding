import json
import numpy as np
from tqdm import tqdm

USER_GRAPH_PATH = "./userGraph/"

with open("./users.json",'r') as load_f:
    items_info = json.load(load_f)

def createUserGraph(uid,item_info,info_type):
    if len(item_info) == 0:
        return 
    
    save_json_path = USER_GRAPH_PATH + info_type + "/" + str(uid) + ".json"
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

    like_info     =  np.array(id_item['like'])
    dislike_info  =  np.array(id_item['dislike'])
    finish_info   =  np.array(id_item['finish'])
    unfinish_info =  np.array(id_item['unfinish'])
    
    createUserGraph(item_id,like_info,"like")
    createUserGraph(item_id,dislike_info,"dislike")
    createUserGraph(item_id,finish_info,"finish")
    createUserGraph(item_id,unfinish_info,"unfinish")