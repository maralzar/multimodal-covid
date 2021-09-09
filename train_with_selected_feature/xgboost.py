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
from sklearn.metrics import mean_squared_error
import numpy as np
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
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.softmax = nn.Softmax(dim=1)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
#         print('-------1',x.shape)
        #conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.bn1(x)
#         print('-------after conv1+maxpool+bn1',x.shape)
        # conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.bn2(x)
#         print('-------after conv2+maxpool+bn2',x.shape)
        # conv3
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.bn3(x)
        
        
        
#         print('-------after conv3+maxpooling+bn3',x.shape)
#         # Pass data through dropout1
#         print('***************************',x.shape)
        conv_out = x.view(-1, self.bs*self.c*64*64*1)
#         # Pass data through fc1
#         print('-------after x.view',conv_out.shape)
        x = self.fc1(conv_out)
#         print('-------after fc1',x.shape)
        x = F.relu(x)
#         x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        
#         print('-------after fc2',x.shape)
#         Apply softmax to x
#         output= self.softmax(x)
#         print('-------fo',output.shape,output)
        return x





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
csv_file = root/'newwww.csv'
print('csv file:',csv_file,'\n','root:',root,'\n')
image_dir = root/'covid_safavi/sample/4173146/lung_white'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
transform_img = transforms.Compose([
    transforms.ToTensor()
])
dset = dataloader.covid_ct(root, csv_file)
train_set, val_set = torch.utils.data.random_split(dset, [250, 47])
print(len(train_set),len(val_set))
train_loader = DataLoader(dataset = train_set, batch_size = 1, shuffle=True, num_workers=0,drop_last=True)
valid_loader = DataLoader(dataset = val_set, batch_size = 1, shuffle=False, num_workers=0,drop_last=True)
Model = Net()
print(Model)
Model = torch.load('./models/epoch_20')
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