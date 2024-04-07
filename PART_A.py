#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from torch.utils.data import Subset
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import albumentations as alb
import argparse
import wandb


# In[2]:
###function for loading the image

def load(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(224,224),cv2.INTER_AREA)
    img=img/255
    return img


# In[3]:
####python class for the model, this model takes in the number of filter, filter size, whether to do batch_norm or not
##, the drop_out probability, activation funtion and number of neuron in the hidden layer as the argument

class CNN_CLASSIFER(nn.Module):
    def __init__(self,num_filter=[3,64,64,64,64,64],filter_size=3,batch_norm=False,drop_out=.2,act=nn.ReLU(),neuron=128):
        super().__init__()
        self.FILTER_NUM=num_filter
        self.filter_size=filter_size
        self.SEQUENTIAL=nn.ModuleList()
        self.activation=act
        self.drop_out=drop_out
        self.batch_norm=batch_norm
        self.neuron=neuron
        for i in range(0,5):
            seq=nn.Sequential(
                nn.Conv2d(self.FILTER_NUM[i],self.FILTER_NUM[i+1],kernel_size=filter_size),
                self.activation,
                nn.MaxPool2d(kernel_size=2)
            )
            if self.batch_norm:
                seq.add_module(f'batch_norm{i+1}',nn.BatchNorm2d(self.FILTER_NUM[i+1]))
            
            self.SEQUENTIAL.append(seq)
        self.drop_OUT=nn.Dropout(p=self.drop_out)
        self.fc1 = nn.Linear(self.FILTER_NUM[-1] * 5 * 5, self.neuron)
        self.fc2 = nn.Linear(self.FILTER_NUM[-1] * 3 * 3, self.neuron)
        self.fc3=nn.Linear(self.neuron,10)
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self,x):
        for i in range(0,5):
            x=self.SEQUENTIAL[i](x)

        x=self.drop_OUT(x)
            
        b,c,w,h=x.shape
        x=x.reshape(b,c*w*h)
        
        
        
        if self.filter_size==3:
            x=self.fc1(x)
        elif self.filter_size==5:
            x=self.fc2(x)
        x=self.fc3(x)
            
        
        
        
        ypred=self.softmax(x)
        
            
        return ypred
        
        


# In[4]:

#### This pytorch lightning used for training the model, this takes in the number of filter, filter size,organisatio
##whether to do batch_norm or not,the drop_out probability, activation funtion and number of neuron in the hidden layer
###, the optimizer as the argument
class CNN_NATURALIST(pl.LightningModule):
    def __init__(self,num_filter=64,filter_size=3,organisation='same',data_aug=False,batch_norm=False,drop_out=.2,
                act='ReLU',log=False,lr=1e-4,neuron=128,optimizer='adam'):
        super().__init__()
        self.num_filter=num_filter
        self.filter_size=filter_size
        self.organisation=organisation
        self.data_aug=data_aug
        self.batch_norm=batch_norm
        self.drop_out=drop_out
        self.act=act
        FILTER_NUM=[self.num_filter]
        self.wandb_log=log
        self.lr=lr
        self.neuron=neuron
        self.optimizer_=optimizer
        for i in range(0,4):
            if self.organisation=='same':
                FILTER_NUM.append(self.num_filter)
            elif self.organisation=='doubling':
                FILTER_NUM.append(int(FILTER_NUM[i]*(2)))
            elif self.organisation=='halving':
                FILTER_NUM.append(int(FILTER_NUM[i]/(2)))
            else:
                raise ValueError('Unidentified organisation')
        FILTER_NUM.insert(0,3)
        self.FILTER_NUM=FILTER_NUM
        
        if self.act=='ReLU':
            self.activation=nn.ReLU()
        elif self.act=='GELU':
            self.activation=nn.GELU()
        elif self.act=='SiLU':
            self.activation=nn.SiLU()
        elif self.act=='Mish':
            self.activation=nn.Mish()
        else:
            raise ValueError(' activation function not found')
        
        self.model=CNN_CLASSIFER(num_filter=FILTER_NUM,filter_size=self.filter_size,batch_norm=self.batch_norm,
                                 drop_out=self.drop_out,act=self.activation,neuron=self.neuron)
        
        self.loss_fun=nn.CrossEntropyLoss()
        if self.optimizer_=='adam':
            self.optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        elif self.optimizer_=='sgd':
            self.optimizer=torch.optim.SGD(self.parameters(),lr=self.lr)
        else:
            raise ValueError('Optimizer is not in the list')
        self.train_loss=[]
        self.train_acc=[]
        self.val_loss=[]
        self.val_acc=[]
        
    def forward(self,x):
        
        ypred=self.model(x)
            
        return ypred
    
    def training_step(self,batch,batch_indx):   ####training happens in this function
        global img
        img,label=batch
        img=img.float()
        if self.data_aug:
            img=[augmentor(img_) for img_ in img]
            img=torch.stack(img)
                
            
        
        ypred=self(img)
        
        loss=self.loss_fun(ypred,label)
        
        accuracy=(torch.argmax(ypred,dim=1)==label).sum()
        
        accuracy=accuracy/len(ypred)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)  ###calculating train loss and accuracy
        self.train_loss.append(loss.cpu().detach().numpy())
        self.train_acc.append(accuracy.cpu().detach().numpy())
        
        return loss
    
    def on_train_epoch_end(self):
        epoch_loss=np.average(self.train_loss)
        epoch_accuracy=np.average(self.train_acc)
        
        
        self.train_loss.clear()                                  ###logging train loss and accuracy in wandb
        self.train_acc.clear()
        
        if self.wandb_log==True:
            wandb.log({"Train_Accuracy":np.round(epoch_accuracy*100,2),"Train_Loss":epoch_loss})
    
        
    def validation_step(self,batch,batch_indx):
        img,label=batch
        img=img.float()
        
        ypred=self(img)
        
        loss=self.loss_fun(ypred,label)
        
        accuracy=(torch.argmax(ypred,dim=1)==label).sum()   ###calculating val loss and accuracy
        
        accuracy=accuracy/len(ypred)
        
        self.log('val_loss', loss,prog_bar=True)
        self.log('val_accuracy', accuracy,prog_bar=True)
        self.val_loss.append(loss.cpu().detach().numpy())
        self.val_acc.append(accuracy.cpu().detach().numpy())
        
        return loss
    
    def on_validation_epoch_end(self):
        epoch_loss=np.average(self.val_loss)
        epoch_accuracy=np.average(self.val_acc)
        
        
        self.val_loss.clear()
        self.val_acc.clear()                 ###logging val loss and accuracy in wandb
        
        if self.wandb_log==True:
            wandb.log({"Val_Accuracy":np.round(epoch_accuracy*100,2),"Val_Loss":epoch_loss,"Epoch":self.current_epoch})
        
        
            
    def configure_optimizers(self):
        return [self.optimizer]


# In[5]:



def train_NN(num_filter=64,filter_size=3,organisation='same',data_aug=False,batch_norm=False,drop_out=.2,
            act='ReLU',log=False,epochs=20,lr=1e-4,neuron=128,optimizer='sgd'):
    model=CNN_NATURALIST(num_filter=num_filter,filter_size=filter_size,organisation=organisation,
                         data_aug=data_aug,batch_norm=batch_norm,drop_out=drop_out,act=act,log=log,lr=lr,neuron=neuron,
                        optimizer=optimizer)
    trainer=pl.Trainer(accelerator='auto',max_epochs=epochs)
    trainer.fit(model,train_loader,val_loader)


# In[11]:


parser = argparse.ArgumentParser()
 
parser.add_argument("-wp", "--wandb_project", default = "myprojectname", help = "Project name used to track experiments ")
parser.add_argument("-we", "--wandb_entity", default = "ee22s060", help = "Wandb Entity ")
parser.add_argument("-a", "--activation", default = "SiLU", choices=['ReLU','GELU','SiLU','Mish'],help = "Activation functions" )
parser.add_argument("-e", "--epochs", default = 20, choices=[10,15,20], help = "Number of epochs to train neural network." , type=int)
parser.add_argument("-b", "--batch_size", default = 16, help = "Batch size ", type=int)
parser.add_argument("-f","--filter_size",default=3,choices=[3,5],help='Filter size',type=int)
parser.add_argument("-n","--num_filter",default=32,choices=[32,64],help='Number of filter',type=int)
parser.add_argument('-bn','--batch_norm',default=True,choices=[True,False],help='Whether to do batch normalization or not')
parser.add_argument('-aug','--augment',default=True,choices=[True,False],help='whether to do data augmentation or not')
parser.add_argument('-d','--drop_out',default=0.3,choices=[0.2,0.3],help='Drop out',type=float)
parser.add_argument('-o','--organisation',default='doubling',choices=['same','doubling','halving'])
parser.add_argument("-lg", "--logs", default = "False", choices = ["True","False"],help = "whether to log or not" )
parser.add_argument('-lr',"--lr",default=1e-4,choices=[1e-3,1e-4,1e-5],help="Learning rate of the model",type=float)
parser.add_argument("-neu","--neuron",default=128,type=int)
parser.add_argument("-opt","--optimizer",default='adam',choices=['sgd','adam'],help="optimizer function")
args = parser.parse_args()

train_path=r'D:\DL_DATA\naturalist\inaturalist_12K\train' ###path for train and test####
test_path=r'D:\DL_DATA\naturalist\inaturalist_12K\val'
transforms=transforms.ToTensor()
train_data_=DatasetFolder(root=train_path,loader=load,extensions='.jpg',transform=transforms) ###loading the train and test data as torch tensors###
test_data=DatasetFolder(root=test_path,loader=load,extensions='.jpg',transform=transforms)
train_indx=[]
val_indx=[]
for i in range(0,len(train_data_),1000): ####taking the indices for train and val so that number of image in each class is same#####
    train_indx+=list(range(i,i+800))
    val_indx+=list(range(i+800,i+1000))
val_indx=val_indx[0:-1]

train_data=Subset(dataset=train_data_,indices=train_indx) 
val_data=Subset(dataset=train_data_,indices=val_indx)

train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,drop_last=True) ###### creating data loader for train,val,test
val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=True,drop_last=True)  
test_loader=DataLoader(test_data,batch_size=args.batch_size,shuffle=True,drop_last=True)

from torchvision import transforms
augmentor=transforms.Compose([
    transforms.RandomHorizontalFlip(p=.2),
    transforms.RandomVerticalFlip(p=.2),      #### using albumentation library for augmentation
    transforms.RandomAutocontrast(p=.2),
    transforms.RandomRotation(degrees=(-10,10)),
    transforms.RandomAdjustSharpness(sharpness_factor=0,p=.2),
    transforms.Resize((224,224))
])
'''
project_name='Fashion_tuning_random'
wandb.login(key="5bfaaa474f16b4400560a3efa1e961104ed54810")
wandb.init(project=args.wandb_project,entity=args.wandb_entity)
'''
parameters=train_NN(num_filter=args.num_filter,filter_size=args.filter_size,organisation=args.organisation
                    ,data_aug=args.augment,batch_norm=args.batch_norm,drop_out=args.drop_out,act=args.activation,
                    log=args.logs,epochs=args.epochs,lr=args.lr,neuron=args.neuron,optimizer=args.optimizer)


# In[ ]:




