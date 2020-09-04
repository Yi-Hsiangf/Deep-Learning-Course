# -*- coding: utf-8 -*-
"""resnet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1meYTw-f9Qxkqa3nWP1AXBBLnIcZrORt6
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from collections import defaultdict
from PIL import Image
import time
import copy
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Model type
MODEL_TYPE = 'RESNET18'

# path
ROOT_PATH = "/content/drive/My Drive/DL_Lab3/data/"
MODEL_PATH = "/content/drive/My Drive/DL_Lab3/model/"

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        
        if self.mode == 'train':
            self.data_transforms = \
            transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.data_transforms = \
            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        # step1.
        path = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(path)

        # step2.
        label = self.label[index]

        # step3. Use the transform.ToTensor() can accomplish two tasks in hint
        # Can also apply more transform on the image.
        img = self.data_transforms(img)

        return img, label


def train_model(model, criterion, optimizer, model_type):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_accs = defaultdict(list)
    
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
        print('-' * 10)

        # save the model as check point
        if epoch%2 == 0:
           check_point = MODEL_PATH + model_type + '_' + MODEL_TYPE + '_' + str(epoch) + '.pkl'
           torch.save(model, check_point)

        for mode in ['train', 'test']:
            # Change the model mode, the model work different
            # in train and test time.
            # (The batch normal, drop out layer work differently)
            if mode == 'train':
                model = model.train()
            else:
                model = model.eval()

            running_loss = 0.0
            running_corrects = 0

            current_batch = 0
            total_batch = math.ceil(dataset_sizes[mode]/BATCH_SIZE)

            for inputs, labels in data_loaders[mode]:
                # print('BATCH {}/{}'.format(current_batch, total_batch))
                current_batch += 1

                # Put the data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # Do the gradient update only on train mode
                with torch.set_grad_enabled(mode == 'train'):
                    # Foward the network
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # If training, do the back propagation and update weight
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
           
            # Compute and print loss, accuracy per epoch
            epoch_loss = running_loss / dataset_sizes[mode]
            epoch_acc = running_corrects.double()/dataset_sizes[mode] * 100
            epoch_accs[mode].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(
                mode, epoch_loss, epoch_acc))

            # deep copy the model
            if mode == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:.2f}%'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, epoch_accs


def plot_acc(epoch_accs, epoch_pretrain_accs):
    epoch_list = [i for i in range(EPOCHS)]
    fig, ax = plt.subplots()  # Create a figure and an axes.

    ax.plot(epoch_list, epoch_accs['train'], label='Train(w/o pretraining)')
    ax.plot(epoch_list, epoch_accs['test'], label='Test(w/o pretraining)')
    
    ax.plot(epoch_list, epoch_pretrain_accs['train'], label='Train(with pretraining)')
    ax.plot(epoch_list, epoch_pretrain_accs['test'], label='Test(with pretraining)')
    
    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('Accuracy%')  # Add a y-label to the axes.
    ax.set_title(MODEL_TYPE)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("accuracy.png")


def plot_confusion(model_path):
    with torch.set_grad_enabled(False):
        eval_model = torch.load(model_path)
        eval_model.to(device)
        eval_model = eval_model.eval()
        class_names = [0, 1, 2, 3, 4]
        
        preds_list = []
        labels_list = []
        for id, (inputs, labels) in enumerate(data_loaders['test']):
            print('id', id)
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            outputs = eval_model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for p in preds:
              preds_list.append(p.item())
            for l in labels:
              labels_list.append(l.item())
    
    print(preds_list)
    print(labels_list)
    confmat = confusion_matrix(y_true=labels_list, y_pred=preds_list, labels=class_names, normalize='true')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=round(confmat[i,j], 2), va='center', ha='center')

    plt.xlabel('predicted label')    
    plt.ylabel('true label')
    plt.savefig("confusion.png")


def evaluate(model_path):
    with torch.set_grad_enabled(False):
        eval_model = torch.load(model_path)
        eval_model.to(device)
        eval_model = eval_model.eval()

        running_corrects = 0
        data_num = 0
        
        # Evaluate 100 epochs
        for id, (inputs, labels) in enumerate(data_loaders['test']):
            # Put the data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Foward the network
            outputs = eval_model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            data_num += len(preds)

    epoch_acc = running_corrects.double()/data_num * 100
    print('Best accuracy: {:.2f}%'.format(epoch_acc.item()))
    print()


data_loaders = {}
dataset_sizes = {}
''' # load the training data
train_data = RetinopathyLoader(ROOT_PATH, mode='train')
dataset_sizes['train'] = len(train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
data_loaders['train'] = train_loader '''

# load the testing data
test_data = RetinopathyLoader(ROOT_PATH, mode='test')
dataset_sizes['test'] = len(test_data)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
data_loaders['test'] = test_loader

# Create model
if MODEL_TYPE == 'RESNET18':
    # Create model for Resenet18
    model = models.resnet18()
    model_pretrain = models.resnet18(pretrained=True)
elif MODEL_TYPE == 'RESNET50':
    # Create model for Resenet50
    model = models.resnet50()
    model_pretrain = models.resnet50(pretrained=True)
else:
    # Invalid model type
    raise Exception('MODEL TYPE INVALID: {}'.format(MODEL_TYPE))

num_ftrs = model.fc.in_features
# Five classes for the dataset [0, 1, 2, 3, 4]
model.fc = nn.Linear(num_ftrs, 5)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

num_ftrs = model_pretrain.fc.in_features
# Five classes for the dataset [0, 1, 2, 3, 4]
model_pretrain.fc = nn.Linear(num_ftrs, 5)
model_pretrain.to(device)
optimizer_pretrain = optim.SGD(model_pretrain.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion_pretrain = nn.CrossEntropyLoss()


if __name__ == '__main__':
    '''  # train the model
    best_model, best_acc, epoch_accs = \
    train_model(model, criterion, optimizer, 'no_pretrained')
    torch.save(best_model, MODEL_PATH + MODEL_TYPE + '_best.pkl') 

    # train the model from pretrain weight
    best_pretrain_model, best_pretrain_acc, epoch_pretrain_accs = \
    train_model(model_pretrain, criterion_pretrain, optimizer_pretrain, 'pretrained')
    torch.save(best_pretrain_model, MODEL_PATH + MODEL_TYPE + '_pretrain_best.pkl')
    
    # print the accuracy
    # print('Best accuracy for ' + MODEL_TYPE + ' without pretrained: {:.2f}%'.format(best_acc.item()))
    # print('Best accuracy for ' + MODEL_TYPE + ' without pretrained: {:.2f}%'.format(max(epoch_accs['test'])))
    print('Best accuracy for pretrained ' + MODEL_TYPE + ' : {:.2f}%'.format(best_pretrain_acc.item()))'''
    
    # plot_acc(epoch_accs, epoch_pretrain_accs)

    model_path = '/content/drive/My Drive/DL_Lab3/model_pretrained_50_10_epochs/RESNET50_pretrain_best.pkl'
    # evaluate(model_path)
    plot_confusion(model_path)