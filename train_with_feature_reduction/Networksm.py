import torch
import torch.nn as nn
import torch.nn.functional as F

class Netm(nn.Module):
    def __init__(self):
        super(Netm, self).__init__()
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
            nn.Conv1d(1,20 , kernel_size=2, stride=1, padding=1),
            nn.ReLU(),nn.BatchNorm1d(num_features=20),
            nn.Conv1d(20, 5, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),nn.BatchNorm1d(num_features=5))

    def forward(self, x,y):
        y = self.layer1(y)
        y = y.view(-1,5*3)
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
        x = self.fc1(conv_out)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x