import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, n_feats, n_samples):
        super(TNet, self).__init__()
        self.n_feats = n_feats
        self.conv1 = nn.Conv1d(n_feats, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_feats*n_feats)
        self.bn1 = nn.LayerNorm(T.Size([64, n_samples]))
        self.bn2 = nn.LayerNorm(T.Size([128, n_samples]))
        self.bn3 = nn.LayerNorm(T.Size([1024,n_samples]))
        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(256)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        batch_size = x.size()[0]
        bias = Variable(torch.from_numpy(np.eye(self.n_feats).flatten().astype(np.float32))).view(1,self.n_feats*self.n_feats).repeat(batch_size,1).to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x + bias
        x = x.view(-1, self.n_feats, self.n_feats)
        return x


class PointNet(nn.Module):
    def __init__(self, mode='classifier', n_samples=1024, n_classes=10):
        super(PointNet, self).__init__()
        self.in_trans = TNet(3, n_samples)
        self.feat_trans = TNet(64, n_samples)
        self.mode = mode
        self.n_samples = n_samples
        self.n_classes = n_classes

        if(self.mode!='classifier' and self.mode!='segmentation'):
            raise Exception("wrong mode selected! choose 'classifier' or 'segmentation'")

        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1),
                                nn.LayerNorm(T.Size([64, n_samples])),
                                nn.ReLU(),
                                nn.Conv1d(64, 64, 1),
                                nn.LayerNorm(T.Size([64, n_samples])),
                                nn.ReLU())

        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1),
                                nn.LayerNorm(T.Size([64, n_samples])),
                                nn.ReLU(),
                                nn.Conv1d(64, 128, 1),
                                nn.LayerNorm(T.Size([128, n_samples])),
                                nn.ReLU(),
                                nn.Conv1d(128, 1024, 1),
                                nn.LayerNorm(T.Size([1024, n_samples])),
                                nn.ReLU())

        self.mlp3 = nn.Sequential(nn.Linear(1024, 512),
                                    nn.LayerNorm(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.LayerNorm(256),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(256, n_classes),
                                    nn.Softmax(1))

        self.mlp4 = nn.Sequential(nn.Conv1d(1088, 512, 1),
                                nn.LayerNorm(T.Size([512, n_samples])),
                                nn.ReLU(),
                                nn.Conv1d(512, 256, 1),
                                nn.LayerNorm(T.Size([256, n_samples])),
                                nn.ReLU(),
                                nn.Conv1d(256, 128, 1),
                                nn.LayerNorm(T.Size([128, n_samples])),
                                nn.ReLU())

        self.mlp5 = nn.Sequential(nn.Conv1d(128, n_classes, 1),
                                nn.LayerNorm(T.Size([n_classes, n_samples])),
                                nn.ReLU(),
                                nn.Softmax(1))

        # self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        input_transform = self.in_trans.forward(T.transpose(x,1,2))
        x = T.bmm(x, input_transform) # nx3

        x = T.transpose(self.mlp1(T.transpose(x,1,2)), 1, 2) # nx64
        feat_transform = self.feat_trans(T.transpose(x,1,2))
        x_to_concat = T.bmm(x, feat_transform) # nx64

        x = T.transpose(self.mlp2(T.transpose(x_to_concat,1,2)), 1, 2) # nx1024
        global_feat = T.max(x, 1, keepdim=True)[0].view(-1, 1024)

        if self.mode == 'classifier':
            x = self.mlp3(global_feat)
            return x, feat_transform

        elif self.mode == 'segmentation':
            x = global_feat.view(-1, 1, 1024).repeat(1, self.n_samples, 1)
            x = T.cat([x_to_concat, x], 2)
            point_feats = T.transpose(self.mlp4(T.transpose(x,1,2)), 1, 2)
            x = T.transpose(self.mlp5(T.transpose(point_feats,1,2)), 1, 2)
            return x