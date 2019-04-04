import torch
import torch.nn.functional as F 
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim

import csv
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

'''
data processing
'''
BATCH_SIZE = 256*16
class FaceLandmarksDataset(data.Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file,datatype='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, iterator=True)
        self.datatype = datatype
        self.csv_file = csv_file
        self.epoch = 1
        self.trainNum = 1406435
        self.testNum = 401838
        
        #Todo 
#         self.landmarks_frame[] # set trainset (0.7)
        
    def __len__(self):
        #print len(self.landmarks_frame)
        #return len(self.landmarks_frame)
        if self.datatype == 'train':
            return int(self.trainNum/BATCH_SIZE)+1
        if self.datatype == 'test':
            return int(self.testNum/BATCH_SIZE)+1
    
    
    def __getitem__(self, idx):
#         print(idx)
        landmarks = self.landmarks_frame.get_chunk(BATCH_SIZE).as_matrix().astype('float')
        # landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
 
        return landmarks
    def refresh(self):
        print("refreshing...")
        self.epoch += 1
        self.landmarks_frame = pd.read_csv(self.csv_file, iterator=True)
        
    def getTrainNum(self):
        return int(self.trainNum/BATCH_SIZE)+1
    
    def getTestNum(self):
        return int(self.testNum/BATCH_SIZE)+1

filename = './trainset.csv'
dataset = FaceLandmarksDataset(filename,'train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

testname = './testset.csv'
testset = FaceLandmarksDataset(testname,'test')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden1 = torch.nn.Linear(n_feature, 128)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(128, 96)
        self.hidden3 = torch.nn.Linear(96, 64)
        self.hidden4 = torch.nn.Linear(64, 48)
        self.hidden5 = torch.nn.Linear(48, 32)
        self.hidden6 = torch.nn.Linear(32, 16)
        self.hidden7 = torch.nn.Linear(16, 8)
        self.out = torch.nn.Linear(8, 2)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden1(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden2(x)) 
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x))      
        x = F.relu(self.hidden5(x)) 
        x = F.relu(self.hidden6(x)) 
        x = F.relu(self.hidden7(x))  

        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        # x = torch.nn.functional.Softmax(x)

        return x
    
    
best_acc = 0
start_epoch = 0
net = Net(n_feature=128*2) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./checkpoint/ckpt.t7')
# net.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']
# print("start_epoch:",start_epoch)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
criterion = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted



def train(epoch):
    print('\ntraining Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    i = 0
    for batch_data in dataloader:
        inputs = batch_data[0][:,1:257]
        inputs = inputs.float()
        targets = batch_data[0][:,257]
        targets = targets.long()
        # print(targets)
  
        if inputs.shape[0] < BATCH_SIZE:
            dataset.refresh()
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
#         outputs = outputs.long()
        loss = criterion(outputs, targets)
        
        # print(targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        
        if i >= dataset.getTrainNum() - 1:
            print('train set:Loss: %.5f | Acc: %.5f%% (%d/%d)'
                % (train_loss/(i+1), 100.*correct/total, correct, total))
        i += 1



def test(epoch):
    print('testing Epoch: %d' % epoch)
    global best_acc
    net.eval() 
    train_loss = 0
    correct = 0
    total = 0
    i = 0
    for batch_data in testloader:
        inputs = batch_data[0][:,1:257]
        inputs = inputs.float()
        targets = batch_data[0][:,257]
        targets = targets.long()
  
        if inputs.shape[0] < BATCH_SIZE:
            testset.refresh()
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        
        if i >= dataset.getTestNum() - 1:
            print('test set:Loss: %.5f | Acc: %.5f%% (%d/%d)'
                % (train_loss/(i+1), 100.*correct/total, correct, total))
        i += 1
    
    # save model
    acc = 100.*correct/total
    if acc > best_acc:
        print('===> epoch ' + str(epoch) + ' Saving model...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    # if epoch == 1000:
    #     print('===> epoch ' + str(epoch) + 'Saving last model...')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/last_ckpt.t7')

for epoch in range(start_epoch,start_epoch+1000):
    train(epoch)
    test(epoch)
