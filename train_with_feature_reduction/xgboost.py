from sklearn.metrics import confusion_matrix

import network as md
import torch
from pathlib import Path
import os
from torchvision import transforms
import dataloader
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from network import Net
from sklearn.metrics import mean_squared_error
import numpy as np


losses = [] 
pred = []
real =[]
valid_losses = []
corrects = 0
training_loss = 0.0
valid_loss = 0.0
valid_correct = 0
root = Path(os.getcwd())
criterion = nn.CrossEntropyLoss()
csv_file = root/'final_with_knn.csv'
print('csv file:',csv_file,'\n','root:',root,'\n')
image_dir = root/'covid_safavi/sample/4173146/lung_white'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
transform_img = transforms.Compose([
    transforms.ToTensor()
])
dset = dataloader.covid_ct(root, csv_file)
train_set, val_set = torch.utils.data.random_split(dset, [250, 71])
print(len(train_set),len(val_set))
train_loader = DataLoader(dataset = train_set, batch_size = 1, shuffle=True, num_workers=0,drop_last=True)
valid_loader = DataLoader(dataset = val_set, batch_size = 1, shuffle=False, num_workers=0,drop_last=True)
Model = Net()
print(Model)
Model = torch.load('./kfold/fold_0/epoch_0')
Model.to(device ='cuda:0')
Model.eval()  
# Optional when not using Model Specific layer
Xtrain = []
Ytrain = []
for i, (data, target,f) in enumerate(train_loader):
    with torch.no_grad():
        print(i,"-------------")
        inputs, labels,features = data,target,f
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = Variable(inputs.view(1,64,512,512,1)).type(torch.FloatTensor).cuda()
#         print(inputs.shape,"inputs")
        outputs = Model(inputs.cuda())
        out=Model.fc2.weight.cpu()
        out = out.reshape(-1)
        f = f.reshape(-1)
        k = np.concatenate((out,f), axis=None)
        Xtrain.append(k)
        Ytrain.append(target)
#xg boost
print(len(Xtrain),len(Ytrain))
X = np.array(Xtrain)
Y = np.array(Ytrain)       
regressor = xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)
regressor.fit(X, Y)
Xtest = []
Ytest = []
Model.eval()  
# Optional when not using Model Specific layer
for i, (data, target,f) in enumerate(valid_loader):
    with torch.no_grad():
        
        inputs, labels,features = data,target,f
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = Variable(inputs.view(1,64,512,512,1)).type(torch.FloatTensor).cuda()
#         print(inputs.shape,"inputs")
        outputs = Model(inputs.cuda())
        out=Model.fc2.weight.cpu()
        out = out.reshape(-1)
        f = f.reshape(-1)
        k = np.concatenate((out,f), axis=None)
        Xtest.append(k)
        Ytest.append(target)
        
X_test = np.array(Xtrain)
Y_test = np.array(Ytrain)
y_pred = regressor.predict(X_test)
print(mean_squared_error(Y_test, y_pred))
txt_xg = open('log/xgboost.txt', 'a')
txt_xg.write('mean square error:'+str(mean_squared_error(Y_test, y_pred)))
txt_xg.close()