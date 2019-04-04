'''
what data set expect to look like:
[128 dimension user vec] [128 dimension item vec] [finish(1) or unfinish(0)]
'''

import pandas as pd
import csv
import json
import numpy as np
from tqdm import tqdm

USER_VEC_PATH = './features/user_finish.csv'
ITEM_VEC_PATH = "./features/item_finish.csv"
SAVING_PATH   = "./dataset.csv"

user_vec = pd.read_csv(USER_VEC_PATH)

user_vec_list = []
for idx in user_vec.ix[:,0]:
    user_vec_list.append(idx)
    
item_vec = pd.read_csv(ITEM_VEC_PATH)

item_vec_list = []
for idx in item_vec.ix[:,0]:
    item_vec_list.append(idx)

out = open(SAVING_PATH,'w', newline='')
csv_write = csv.writer(out)

rows = []
for i in range(128*2):
    temp_str = "x_" + str(i)
    rows.append(temp_str)
    
    
rows.append("label")
csv_write.writerow(rows)

def saveDateToLocal(user_vec, item_vec, label):
    # check dimension
    if len(user_vec)!= 128 or len(item_vec)!= 128:
        print("dimension error!")
        return 
    
    # concat user_vec and item_vec
    data = user_vec + item_vec
    data.append(label)
#     print(len(data))
    # save to dataset.csv
    csv_write.writerow(data)

def getItemVec(item_index):
    item_vec_dataFrame = item_vec.ix[item_index,1:]
    itemVec = []
    for i in range(128):
        index = "x_" + str(i)
        itemVec.append(round(item_vec_dataFrame[index],4))
    
    return itemVec

def getUserVec(uid_index):
    
    user_vec_dataFrame = user_vec.ix[uid_index,1:]
    userVec = []
    for i in range(128):
        index = "x_" + str(i)
        userVec.append(round(user_vec_dataFrame[index],4))
    
    return userVec

'''
create data set through users.json
'''
with open("./users.json",'r') as load_f:
    users_info = json.load(load_f)
for key in users_info.keys():
    users_info[key].pop('like')
    users_info[key].pop('dislike')

i = 0
for uid in tqdm(users_info.keys()):
    id_user = users_info.get(str(uid),{})
    finish_info   =  np.array(id_user['finish'])
    unfinish_info =  np.array(id_user['unfinish'])
    
    # get user_vec through uid
    if int(uid) not in user_vec_list:
        continue
        
    uid_index = user_vec_list.index(int(uid))
    temp_user_vec = getUserVec(uid_index)
#     print(uid_index,temp_user_vec)
    
    for item_id in finish_info:
        if int(item_id) not in item_vec_list:
            continue
        item_index = item_vec_list.index(int(item_id))
        temp_item_vec = getItemVec(item_index)
#         print(temp_item_vec)
        saveDateToLocal(temp_user_vec, temp_item_vec, 1)

    for item_id in unfinish_info:
        if int(item_id) not in item_vec_list:
            continue
        item_index = item_vec_list.index(int(item_id))
        temp_item_vec = getItemVec(item_index)
        saveDateToLocal(temp_user_vec, temp_item_vec, 0)