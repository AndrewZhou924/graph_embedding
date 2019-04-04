import pandas as pd
import numpy as np
from sklearn.utils import shuffle


dataset = pd.read_csv('./dataset.csv')
dataset = shuffle(dataset)
print(dataset.shape)

num = dataset.shape[0]
# print(num)

trainset = dataset[:int(num*0.7)]
testset = dataset[int(num*0.7):int(num*0.9)]
valset = dataset[int(num*0.9):]


trainset.to_csv('trainset.csv')
testset.to_csv('testset.csv')
valset.to_csv('valset.csv')