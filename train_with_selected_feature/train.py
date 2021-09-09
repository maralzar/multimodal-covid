
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
from network import Net
print(logging.handlers)
logger = logging.getLogger()
fh = logging.handlers.RotatingFileHandler('./logfile_train_with_selected_feature.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.warning('This will get logged to a file')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main(epoch_num=10,learning_rate = 0.01,kfold_num =5,load_last_model=False):
    root = Path(os.getcwd())
    logger.warning(str(root))
    csv_file = root/'ExtraTreesClassifier_feature_selection.csv'
    print('csv file:',csv_file,'\n','root:',root,'\n')
    print(root)
    transform_img = transforms.Compose([
        transforms.ToTensor()])
    dset = dataloader.covid_ct(root, csv_file)

    Model = Net()
    Model.to(device ='cuda:0')
#     if load_last_model:
#         Model = torch.load('./models/epoch_20')
    print("load model")	
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.000001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(),lr = learning_rate)
    num_epochs = 100
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
        valid_loader = DataLoader(dataset = val_set, batch_size = 1, shuffle=False, num_workers=0,drop_last=True)
        
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
                        print('----------------iter : ',i)
                        logger.warning("------- iteration phase train:%d ----------"%i)
                        inputs, labels = data,target
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # print(labels)
                        inputs = Variable(inputs.view(4,64,512,512,1)).type(torch.FloatTensor)
                        targets = Variable(labels).type(torch.LongTensor).cuda()
                        optimizer.zero_grad()
                        Model.train()
                        outputs = Model(inputs.cuda())
                        targets = torch.squeeze(targets, 0)
                        loss = criterion(outputs.cuda(),targets.cuda())
                        losses += [loss.item()]
                        loss.backward()
                        optimizer.step()
                        corrects += (torch.argmax(outputs,dim=1) == targets.cuda()).sum().item()
                        training_loss += loss.item()
                        logger.warning('\nEpoch [%2d/%2d], Step [%3d/%3d], Loss train: %.4f '
                                % (epoch + 1, num_epochs, i + 1, len(train_loader) , loss.item()))
                        
                
                       
        
                        
            
                if epoch % 5 == 0:
                        print('Corrects in epoch_%d: %d from %d' %(epoch+1 ,corrects,4*len(train_loader)))
                        print('Train Accuracy in epoch_%d: %.4f, Train Loss: ' 
                        %(epoch+1 ,(100 * (corrects / len(train_loader)))),training_loss/len(train_loader)) 
                        logger.warning('Corrects  train in epoch_%d: %d from %d' %(epoch+1 ,corrects,4*len(train_loader)))
                        logger.warning('Train Accuracy in epoch_%d: %.4f, Train Loss total: %.4f' %(epoch+1 ,(100 * (corrects / (4*len(train_loader)))),training_loss/len(train_loader)))
                        torch.save(Model,str(root/'kfold')+'/fold_'+str(fold)+'/epoch_'+str(epoch)) 
                
                        Model.eval()  
                        logger.warning('-----------------------evaluation-----------------------')
                        print("model eval:")
                        for i, (data, target,features) in enumerate(valid_loader):
                                
                                print('--------eval--------iter = ',i)
                                
                                with torch.no_grad():
                                        inputs, labels,f = data,target,features
                                        inputs = inputs.to(device)
                                        labels = labels.to(device)
                                        inputs = Variable(inputs.view(1,64,512,512,1)).type(torch.FloatTensor).cuda()
                                        targets = Variable(labels).type(torch.LongTensor).cuda()
                                        outputs = Model(inputs.cuda())
                                        pred.append(torch.argmax(outputs,dim=1).cpu().numpy()[0])
                                        real.append(targets.cpu().numpy()[0])
                                        valid_loss += criterion(outputs,targets)
                                        valid_correct += (torch.argmax(outputs,dim=1) == targets).sum().item()
                                        print(valid_correct)
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
