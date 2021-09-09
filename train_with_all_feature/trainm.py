
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import os
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import PIL
from pathlib import Path
from PIL import Image
from torchvision.models import vgg19_bn
import glob
from torch.autograd import Variable
import network as md
import dataloader
from torch.utils.data import Dataset, DataLoader
random.seed(0)
from sklearn.model_selection import KFold
import pathlib
import pandas as pd
import logging
import logging.handlers
# print(logging.handlers)
logger = logging.getLogger()
fh = logging.handlers.RotatingFileHandler('./logfile_train_with_all_feature_miyani.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.warning('This will get logged to a file')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c= 8
        self.bs = 4
        self.conv1 = nn.Conv3d(64,self.c, kernel_size=(3,3,1),stride=1,padding=1,bias=False)
        self.conv2 = nn.Conv3d(self.c,self.c*2,kernel_size=(3,3,1),stride=1,padding=1,bias=False)
        self.conv3 = nn.Conv3d(self.c*2,self.c*4,kernel_size=(3,3,1),stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm3d(self.c, affine = False)
        self.bn2 = nn.BatchNorm3d(self.c*2, affine = False)
        self.bn3 = nn.BatchNorm3d(self.c*4, affine = False)
        self.dropout1 = nn.Dropout3d(0.25)
        self.fc1 = nn.Linear((self.bs*self.c*64*64*1), 256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 60, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),nn.BatchNorm1d(num_features=60),
            nn.Conv1d(60, 30, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),nn.BatchNorm1d(num_features=30))

    def forward(self, x,y):
        # Pass data through conv1
        # print('-------1',y.shape)
        #conv1

        y = self.layer1(y)
        y = y.view(-1,30*69)
        # print("y.shape",y.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.bn3(x)
        conv_out = x.view(-1, self.bs*self.c*64*64*1)
        c = torch.cat([conv_out, y], dim=1)
        # print(c.shape)
#         # Pass data through fc1

        # print('-------after x.view',c.shape)
        x = self.fc1(conv_out)
#         print('-------after fc1',x.shape)
        x = F.relu(x)
#         x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x


def main(epoch_num=50,learning_rate = 0.000001,kfold_num =5):
    root = Path(os.getcwd())
    logger.warning(str(root))
    csv_file = root/'final_with_knn.csv'
#     print('csv file:',csv_file,'\n','root:',root,'\n')
#     print(root)
 
    

    transform_img = transforms.Compose([
        transforms.ToTensor()
])
    dset = dataloader.covid_ct(root, csv_file)

    Model = Net()
    Model.to(device ='cuda:0')
#     Model = torch.load('./models/epoch_20')
    print("load model")	
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(),lr = learning_rate)
    #for cross entropy, 2 neuron output
    num_epochs = epoch_num
    losses = [] 
    valid_losses = []
    corrects = 0
    training_loss = 0.0
    #make folder for each fold
    file = pathlib.Path(str(root/'kfold/fold_')+'0')
    if not(file.exists ()):
        for num_folder  in range(kfold_num):
                #     print(str(root)+'/kfold/fold_'+str(num_folder))
                os.mkdir(str(root/'kfold/fold_')+str(num_folder))
    
    for  fold in range(0,kfold_num):
        logger.warning(str('fold: %d'%fold))
        train_index =pd.read_csv('./dataset_index/fold'+str(fold)+'.csv')
        train_index.columns = ["index"]
        train_index_final=train_index['index'].values.tolist()
        # print(train_index_final)
        train_index_final = [int(a_) for a_ in train_index_final]
        test_index =pd.read_csv('./test_index/fold'+str(fold)+'.csv')
        test_index.columns = ["index"]
        test_index_final = test_index['index'].values.tolist()
        test_index_final = [int(a_) for a_ in test_index_final]
        train_set = torch.utils.data.dataset.Subset(dset,train_index_final)
        val_set = torch.utils.data.dataset.Subset(dset,test_index_final)
        train_loader = DataLoader(dataset = train_set, batch_size = 4, shuffle=True, num_workers=0,drop_last=True)
        valid_loader = DataLoader(dataset = val_set, batch_size = 4, shuffle=False, num_workers=0,drop_last=True)

        print(len(train_loader),len(valid_loader))
        for epoch in range(num_epochs):
                print("-------------------- epoch %d ------------------\n" %epoch)
                logger.warning("-------------------- epoch %d ------------------\n" %epoch)
                corrects=0
                training_loss = 0.0 
                valid_loss = 0.0
                valid_correct = 0
                pred = []
                real =[]
                valid_losses = []
                for i, (data, target,feature) in enumerate(train_loader):
                        print('----------------iter  = ',i)
                        inputs, labels = data,target
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # print(labels)
                        inputs = Variable(inputs.view(4,64,512,512,1)).type(torch.FloatTensor)
                        targets = Variable(labels).type(torch.LongTensor).cuda()
                        f = Variable(feature).type(torch.FloatTensor).cuda()
                        optimizer.zero_grad()
                        Model.train()
                        # print(f.shape)
                        outputs = Model(inputs.cuda(),f.cuda())
                        targets = torch.squeeze(targets, 0)
                        loss = criterion(outputs.cuda(),targets.cuda())
                        losses += [loss.item()]
                        loss.backward()
                        optimizer.step()
                        # print("outputs: ",torch.argmax(outputs,dim=1) ,"targets: ",targets.cuda())
                        corrects += (torch.argmax(outputs,dim=1) == targets.cuda()).sum().item()
                        # print(corrects)
                        training_loss += loss.item()
                        logger.warning('\nEpoch [%2d/%2d], Step [%3d/%3d], Loss train: %.4f '
                                % (epoch + 1, num_epochs, i + 1, len(train_loader) , loss.item()))
                        break
                       
        
                        
             
                if epoch % 5 == 0:
                        print('Corrects in epoch_%d: %d from %d' %(epoch+1 ,corrects,4*len(train_loader)))
                        print('Train Accuracy in epoch_%d: %.4f, Train Loss: ' 
                        %(epoch+1 ,(100 * (corrects / len(train_loader)))),training_loss/len(train_loader)) 
                        logger.warning('Corrects  train in epoch_%d: %d from %d' %(epoch+1 ,corrects,4*len(train_loader)))
                        logger.warning('Train Accuracy in epoch_%d: %.4f, Train Loss total: %.4f' %(epoch+1 ,(100 * (corrects / (4*len(train_loader)))),training_loss/len(train_loader)))
                        torch.save(Model,str(root/'kfold')+'/fold_'+str(fold)+'/epoch_'+str(epoch)) 
                
                        Model.eval()  
                        # Optional when not using Model Specific layer
                        logger.warning('-----------------------evaluation-----------------------')
                        for i, (data, target,features) in enumerate(valid_loader):
                                print('----------------iter = ',i)
                                with torch.no_grad():
                                        inputs, labels,f = data,target,features
                                        inputs = inputs.to(device)
                                        labels = labels.to(device)
                                        inputs = Variable(inputs.view(4,64,512,512,1)).type(torch.FloatTensor).cuda()
                                        targets = Variable(labels).type(torch.LongTensor).cuda()
                                        f = Variable(feature).type(torch.FloatTensor).cuda()
                                        outputs = Model(inputs.cuda(),f.cuda())
                                        pred.append(torch.argmax(outputs,dim=1).cpu().numpy()[0])
                                        real.append(targets.cpu().numpy()[0])
                                        valid_loss += criterion(outputs,targets)
                                        valid_correct += (torch.argmax(outputs,dim=1) == targets).sum().item()
                                        valid_losses += [valid_loss.item()]
                        print('Validation Accuracy in epoch: %.4f, Valid Loss: %.4f'
                                %((100 * (valid_correct / len(val_set))),valid_loss/len(val_set)))
                        
                        logger.warning(str('Validation Accuracy in epoch_%d:  %.4f ,Valid Loss: %.4f'%(epoch,(100 * (valid_correct / len(val_set))),np.mean(valid_losses))))
                        tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
                        logger.warning('fold: '+str(fold)+'tn: '+str(tn)+'fp: '+str(fp)+'fn: '+str(fn)+'tp: '+str(tp))              
                        logger.warning(str(confusion_matrix(real, pred)))
                        logger.warning('Finished evalating')
                

if __name__ == "__main__":
    main(kfold_num =5)
