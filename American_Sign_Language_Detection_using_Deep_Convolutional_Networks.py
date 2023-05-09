
# Importing the required libraries

import os
import numpy as np
import matplotlib.pyplot as plt 
from dataloader import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torch.nn.modules.dropout import Dropout
import torchvision
from torchsummary import summary
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

# Defining the train transforms

train_transform=transforms.Compose([transforms.ColorJitter(brightness=0.3, 
                          saturation=0.1, contrast=0.1),transforms.ToTensor()])

Train_data_path="../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
train_data=ImageFolder(Train_data_path,transform=train_transform)
len(train_data)

# Initializing the train dataloader

train_loader=train_data_loader(train_data,100)
classes=train_loader.dataset.classes

# Visualizing 5 images in the train data and checking the size

for i in range(5):
    for img, label in train_loader:
        print("Image size is {0}".format(img.shape))
        print('Ground truth', classes[label[0]])
        plt.imshow(img[0].permute(1, 2, 0))
        plt.show()
        break

# #### Defining Neural Network architecture to train the model

# #### 2 CNN hidden layers

class Neural_Net_1(nn.Module):
    def __init__(self):
        super(Neural_Net_1,self).__init__()
        self.NN_1_feature_extract=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16,32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2)
        )

        self.classify=nn.Sequential(
            nn.Linear(32*100*100,80),
            nn.Dropout(0.2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.2),
            nn.Linear(80,29),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat_extract=self.NN_1_feature_extract(x)
        feat_extract=feat_extract.view(-1,32*100*100)
        classification=self.classify(feat_extract)
        return classification

# Verifying the model parameters

Model_1=Neural_Net_1()
Model_1.cuda()

# Setting up the loss function and optimizer

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(Model_1.parameters(),lr=0.01,weight_decay=0.0001)

# Checking for availabity of GPU
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device in use is {}".format(device)) 


# Training the models

def Model_train(number_of_epochs, train_loader, Model, loss_function, optimizer):
    count=0
    for epoch in range(number_of_epochs):
        correct=0
        for images, labels in train_loader:
            count+=1
            images = images.cuda()
            labels = labels.cuda()
            outputs=Model(images)
            loss=loss_function(outputs, labels)
            # Back Propogation
            optimizer.zero_grad()
            loss.backward()
            # Update Weights (Optimize the model)
            optimizer.step()
            # Checking the performance
            predictions=torch.max(outputs,1)[1]
            correct+=(predictions==labels).cpu().sum().numpy()
        print("Epoch is: {0}, Loss is {1} and Accuracy is: {2}".format(epoch+1,loss.data,100*correct/len(train_loader.dataset)))

    print("Training finished")

Model_train(10, train_loader,Model_1,loss_function,optimizer)


# #### Single layer CNN

class Neural_Net_2(nn.Module):
    def __init__(self):
        super(Neural_Net_2,self).__init__()
        self.NN_2_feature_extract=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=(5,5),padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
        )

        self.classify=nn.Sequential(
            nn.Linear(160000,128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128,29),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat_extract=self.NN_2_feature_extract(x)
        feat_extract=feat_extract.view(-1,100*100*16)
        classification=self.classify(feat_extract)
        return classification


Model_2=Neural_Net_2()
Model_2.cuda()

loss_function_2=nn.CrossEntropyLoss()
optimizer_2=torch.optim.Adam(Model_2.parameters(),lr=0.01,weight_decay=0.0001)

Model_train(10, train_loader,Model_2,loss_function_2,optimizer_2)

class Neural_Net_3(nn.Module):
    def __init__(self):
        super(Neural_Net_3,self).__init__()
        self.NN_2_feature_extract=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=(5,5),padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),

            nn.Conv2d(64,128,kernel_size=(5,5),padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
            
            nn.Conv2d(128,256,kernel_size=(5,5),padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
        )

        self.classify=nn.Sequential(
            nn.Linear(256*25*25,256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256,29),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat_extract=self.NN_2_feature_extract(x)
        feat_extract=feat_extract.view(-1,25*25*256)
        classification=self.classify(feat_extract)
        return classification

Model_3=Neural_Net_3()
Model_3.cuda()

class Neural_Net_4(nn.Module):
    def __init__(self):
        super(Neural_Net_4,self).__init__()
        self.NN_4_feature_extract=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=(3,3),padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),

            nn.Conv2d(64,128,kernel_size=(3,3),padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
            
            nn.Conv2d(128,256,kernel_size=(3,3),padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
        )

        self.classify=nn.Sequential(
            nn.Linear(256*25*25,128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128,29),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat_extract=self.NN_4_feature_extract(x)
        feat_extract=feat_extract.view(-1,25*25*256)
        classification=self.classify(feat_extract)
        return classification

Model_4=Neural_Net_4()
Model_4.cuda()

# As accuracy is increasing with increase in No.of hidden convolution layers, performing hyperparameter tuning only for 3 CNN hidden layers with change in learning rate and Kernel size

#  Kernel size (5 x 5)
learning_rates=[0.01,0.001]

for l_rate in learning_rates:
    loss_function_CNN=nn.CrossEntropyLoss()
    optimizer_CNN=torch.optim.Adam(Model_3.parameters(),lr=l_rate,weight_decay=0.0001)
    Model_train(7, train_loader,Model_3,loss_function_CNN,optimizer_CNN)

# Kernel size (3 x 3)

for l_rate in learning_rates:
    loss_function_CNN_2=nn.CrossEntropyLoss()
    optimizer_CNN_2=torch.optim.Adam(Model_4.parameters(),lr=l_rate,weight_decay=0.0001)
    Model_train(7, train_loader,Model_4,loss_function_CNN_2,optimizer_CNN_2)

# ### Training the ResNet50 Model by varying learning rates

resnet_model = torchvision.models.resnet50(pretrained=True)
resnet_model

for param in resnet_model.parameters():
    param.requires_grad = False

features_inp = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(features_inp, 29)

resnet_model.to(device)
summary(resnet_model, (3, 200, 200), batch_size=100)


for l_rate in learning_rates:
    loss_function_resnet=nn.CrossEntropyLoss()
    optimizer_resnet=torch.optim.Adam(resnet_model.parameters(),
                                      lr=l_rate,weight_decay=0.0001)
    Model_train(7, train_loader,resnet_model,
                loss_function_resnet,optimizer_resnet)

# Training the ResNet18 Model by varying learning rates

resnet_18 = torchvision.models.resnet18(pretrained=True)
for param in resnet_18.parameters():
    param.requires_grad = False

features_inp = resnet_18.fc.in_features
resnet_18.fc = torch.nn.Linear(features_inp, 29)

resnet_18.to(device)
summary(resnet_18, (3, 200, 200), batch_size=100)

for l_rate in learning_rates:
    loss_function_res_18=nn.CrossEntropyLoss()
    optimizer_res_18=torch.optim.Adam(resnet_18.parameters(),
                                      lr=l_rate,weight_decay=0.0001)
    Model_train(5, train_loader,resnet_18,
                loss_function_res_18,optimizer_res_18)
    
# Training the Alexnet Model by varying learning rates

alexnet = torchvision.models.alexnet(pretrained=True)
for param in alexnet.parameters():
    param.requires_grad = False

features_inp = alexnet.fc.in_features
alexnet.fc = torch.nn.Linear(features_inp, 29)

alexnet.to(device)
summary(alexnet, (3, 200, 200), batch_size=100)

for l_rate in learning_rates:
    loss_function_alex=nn.CrossEntropyLoss()
    optimizer_alex=torch.optim.Adam(alexnet.parameters(),
                                      lr=l_rate,weight_decay=0.0001)
    Model_train(5, train_loader,alexnet,
                loss_function_alex,optimizer_alex)


# From all the trained models, ResNet18 gave the best performance on the training set. So, using that trained model to evaluate the test set performance

# Also, apart from the state-of-the-art ResNet18 model, verifying the test set performance on best model architecture other than ResNet18

test_filepath = "../input/asl-alphabet/asl_alphabet_test/"
test_transforms = transforms.Compose([
    transforms.ToTensor()])

test_dataset = torchvision.datasets.ImageFolder(test_filepath, transform=test_transforms)
print("Test Dataset Info:\n",test_dataset)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1)

# Getting the test labels

test_filepath = "../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/"
labels_map = {'A':0,'B':1,'C': 2, 'D': 3, 'E':4,'F':5,'G':6, 'H': 7, 'I':8, 'J':9,'K':10,'L':11, 'M': 12, 'N': 13, 'O':14, 
                'P':15,'Q':16, 'R': 17, 'S': 18, 'T':19, 'U':20,'V':21, 'W': 22, 'X': 23, 'Y':24, 'Z':25, 
                'del': 26, 'nothing': 27,'space':28}
test_labels = []
for folder_name in os.listdir(test_filepath):
    label = folder_name.replace("_test.jpg","")
    label = labels_map[label]
    test_labels.append(np.array(label))
test_labels.sort()

# Predicting the test set labels using CNN with 3 convolutional layers and kernel width of (3 x 3)

pred_test=[]
test_labels_list=[]
with torch.no_grad():
    correct = 0
    for (images,_),labels in zip(test_dataloader,test_labels):
        Model_4.eval()
        images = images.to(device)
        output = Model_4(images)
        prediction = torch.max(output,1)[1]
        pred_test.append(prediction.cpu().numpy()[0])
        correct += (prediction.cpu().numpy()[0] == labels)
        test_labels_list.append(labels)
    print("Accuracy :",(correct/len(test_dataloader.dataset))*100,"%")

confusion_matrix_test=confusion_matrix(pred_test, test_labels_list)

plt.figure(figsize=(16,12))
sns.heatmap(confusion_matrix_test, annot=True)
plt.show()

# Mean F1 score with 3 Convolutional Layers

np.mean(f1_score(pred_test,test_labels_list, average=None))

# Predicting the test set labels using ResNet18 model

pred_test=[]
test_labels_list=[]
with torch.no_grad():
    correct = 0
    for (images,_),labels in zip(test_dataloader,test_labels):
        resnet_18.eval()
        images = images.to(device)
        output = resnet_18(images)
        prediction = torch.max(output,1)[1]
        pred_test.append(prediction.cpu().numpy()[0])
        correct += (prediction.cpu().numpy()[0] == labels)
        test_labels_list.append(labels)
    print("Accuracy :",(correct/len(test_dataloader.dataset))*100,"%")

confusion_matrix_test=confusion_matrix(pred_test, test_labels_list)

plt.figure(figsize=(16,12))
sns.heatmap(confusion_matrix_test, annot=True)
plt.show()

np.mean(f1_score(pred_test,test_labels_list, average=None))


