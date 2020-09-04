import numpy as np
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

EPOCH = 100
LR = 0.001
BATCH_SIZE = 32


def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label


class EGGnet(torch.nn.Module):
    def __init__(self):
        super(EGGnet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = (1, 51),
                stride = (1, 1),
                padding = (0, 25),
                bias = False
            ),
            nn.BatchNorm2d(
                16,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = (2, 1),
                stride = (1, 1),
                groups=16,
                bias = False
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=0
            ),
            nn.Dropout(p=0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = (1, 15),
                stride = (1, 1),
                padding=(0, 7),
                bias = False
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            ),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(
                in_features=736,
                out_features=2,
                bias=True
            )
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        
        x = x.view(-1, 736)
        output = torch.sigmoid(self.classify(x))
        return output


def train(loader):
    for epoch in range(EPOCH):
        for step, (data, label) in enumerate(loader):
            data = data.cuda()
            label = label.cuda() 
            output = Model(data.float())
            loss = loss_func(output, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss.data = loss.data.cpu()
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            
Model = EGGnet()
Model.cuda()
optimizer = torch.optim.Adam(Model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = read_bci_data()
    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    train_dataset = Data.TensorDataset(train_data, train_label)

    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )

    print(Model)
    Model = Model.float()
    train(train_loader)
    
    


