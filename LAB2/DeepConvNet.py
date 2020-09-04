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

EPOCH = 300
LR = 0.001
BATCH_SIZE = 64


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


class DeepConvNet(torch.nn.Module):
    def __init__(self, activation_type="ELU"):
        super(EGGnet, self).__init__()
        if activation_type == "ELU":
            activation_function = nn.ELU(alpha=1.0)
        elif activation_type == "ReLU":
            activation_function = nn.ReLU()
        elif activation_type == "LReLU":
            activation_function = nn.LeakyReLU()

        self.conv1 = nn.Conv2d
        
        
    
    def forward(self, x):
        

def train(Model, optimizer, train_loader, test_loader):
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(EPOCH):
        train_total = 0
        train_correct = 0
        for step, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda() 
            output = Model(data.float())
            loss = loss_func(output, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print accuracy of train data
            _, predicted = torch.max(output, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()
       
            
        accuracy = 100 * train_correct / train_total
        train_accuracy_list.append(accuracy)
        print('Train Accuracy: %.2f %%' % accuracy)
        
        loss.data = loss.data.cpu()
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
        
        # Print accuracy of test data
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for step, (data, label) in enumerate(test_loader):
                data = data.cuda()
                label = label.cuda()
                output = Model(data.float())

                # Print accuracy of train data
                _, predicted = torch.max(output, 1)
                test_total += label.size(0)
                test_correct += (predicted == label).sum().item()

        accuracy = (100 * test_correct / test_total)
        test_accuracy_list.append(accuracy)
        print('Test Accuracy: %.2f %%' % accuracy)
        print()
    return train_accuracy_list, test_accuracy_list     


def plot_accuracy(ELU_train, ELU_test, ReLU_train, ReLU_test, LReLU_train, LReLU_test):
    epoch_list = [i for i in range(EPOCH)]
    fig, ax = plt.subplots()  # Create a figure and an axes.

    ax.plot(epoch_list, ELU_train, label='ELU_train')
    ax.plot(epoch_list, ELU_test, label='ELU_test')

    ax.plot(epoch_list, ReLU_train, label='ReLU_train')
    ax.plot(epoch_list, ReLU_test, label='ReLU_test')

    ax.plot(epoch_list, LReLU_train, label='Leaky_ReLU_train')
    ax.plot(epoch_list, LReLU_test, label='Leaky_ReLU_test')

    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('Accuracy%')  # Add a y-label to the axes.
    ax.set_title("EGGnet")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("accuracy.png")


# Declare Models 
ModelELU = EGGnet("ELU")
ModelELU.cuda()
ModelELU = ModelELU.float()
# Declare Optimizer
optimizerELU = torch.optim.Adam(ModelELU.parameters(), lr=LR)

ModelReLU = EGGnet("ReLU")
ModelReLU.cuda()
ModelReLU = ModelReLU.float()
# Declare Optimizer
optimizerReLU = torch.optim.Adam(ModelReLU.parameters(), lr=LR)

ModelLReLU = EGGnet("LReLU")
ModelLReLU.cuda()
ModelLReLU = ModelLReLU.float()
# Declare Optimizer
optimizerLReLU = torch.optim.Adam(ModelLReLU.parameters(), lr=LR)

# Declare loss function
loss_func = nn.CrossEntropyLoss()


if __name__ == '__main__':
    # Load training data
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

    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)
    test_dataset = Data.TensorDataset(test_data, test_label)
    
    # Load 
    test_loader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )
    
    
    print('Train Model with activation function ELU')
    print(ModelELU)
    ELU_train, ELU_test = train(ModelELU, optimizerELU, train_loader, test_loader)
    
    print('Train Model with activation function ReLU')
    print(ModelReLU)
    ReLU_train, ReLU_test = train(ModelReLU, optimizerReLU, train_loader, test_loader)
    
    print('Train Model with activation function Leaky ReLU')
    print(ModelLReLU)
    LReLU_train, LReLU_test = train(ModelLReLU, optimizerLReLU, train_loader, test_loader)
    
    plot_accuracy(ELU_train, ELU_test, ReLU_train, ReLU_test, LReLU_train, LReLU_test)
    
    highest_test_accuracy = max(ELU_test + ReLU_test + LReLU_test)
    print("Highest Test Accuracy: %.2f %%" % highest_test_accuracy)
    
    


