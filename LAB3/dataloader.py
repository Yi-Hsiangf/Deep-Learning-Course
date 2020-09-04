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
ROOT_PATH = "D:/user/Desktop/DL_Lab3/data/data/"

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

        self.transformations = transforms.Compose([transforms.ToTensor()])
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
        img = self.transformations(img)

        return img, label


def train_model(model, criterion, optimizer, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_accs = defaultdict(list)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
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

            for inputs, labels in data_loaders[mode]:
                # Put the data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # Do the gradient update only on train mode
                with torch.set_grad_enabled(mode == 'train'):
                    # Foward the network
                    outputs = model(inputs)
                    _, preds = torch.max(output, 1)
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
            epoch_acc = running_corrects.double() / dataset_sizes[mode]
            epoch_accs[mode].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if mode == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, epoch_accs


def plot_accuracy(epoch_accs):
    epoch_list = [i for i in range(EPOCH)]
    fig, ax = plt.subplots()  # Create a figure and an axes.

    ax.plot(epoch_list, epoch_accs['train'], label='Train(w/o pretraining)')
    ax.plot(epoch_list, epoch_accs['test'], label='Test(w/o pretraining)')

    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('Accuracy%')  # Add a y-label to the axes.
    ax.set_title("ResNet18")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("accuracy.png")


data_loaders = {}
dataset_sizes = {}
# load the training data
train_data = RetinopathyLoader(ROOT_PATH, mode='train')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
data_loaders['train'] = train_loader
dataset_sizes['train'] = len(train_data)

# load the testing data
test_data = RetinopathyLoader(ROOT_PATH, mode='test')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
data_loaders['test'] = test_loader
dataset_sizes['test'] = len(test_data)

# Create model for Resenet18
model_resnet18 = models.resnet18()
num_ftrs = model_resnet18.fc.in_features
# Five classes for the dataset [0, 1, 2, 3, 4]
model_resnet18.fc = nn.Linear(num_ftrs, 5)
model_resnet18.to(device)
optimizer_resnet18 = optim.SGD(model_resnet18.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion_resnet18 = nn.CrossEntropyLoss()


if __name__ == '__main__':
    # train the model for ResNet18
    best_model, best_acc, epoch_accs = \
    train_model(model_resnet18, criterion_resnet18, optimizer_resnet18, num_epochs=EPOCHS)

    plot_acc(epoch_accs)


