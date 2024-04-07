#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse


# In[2]:

###function for loading the image
def load(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(224,224),cv2.INTER_AREA)
    img=img/255
    return img


# In[3]:


class CNN_NATURALIST(pl.LightningModule):
    def __init__(self,data_aug=False,lr=1e-4):
        super().__init__()
        self.data_aug=data_aug
        self.lr_rate=lr
        
        model=torchvision.models.vgg19(pretrained=True)
        for params in model.features[0:15].parameters():
            params.requires_grad=False
        self.model=model
        self.model.classifier[6]=nn.Linear(4096,10)
        
        self.loss_fun=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr_rate)
        
        self.train_loss=[]
        self.train_acc=[]
        self.val_loss=[]
        self.val_acc=[]
        
    def forward(self,x):
        
        ypred=self.model(x)
        #ypred=nn.Softmax(dim=1)(ypred)
            
        return ypred
    
    ####training happens in this function
    def training_step(self,batch,batch_indx):
        img,label=batch
        img=img.float()
        if self.data_aug:
            img=[augmentor(img_) for img_ in img]
            img=torch.stack(img)
                
            
        
        ypred=self(img)
        
        loss=self.loss_fun(ypred,label)
        
        accuracy=(torch.argmax(ypred,dim=1)==label).sum()              
        
        accuracy=accuracy/len(ypred)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)      ###calculating train loss and accuracy
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss.append(loss.cpu().detach().numpy())
        self.train_acc.append(accuracy.cpu().detach().numpy())
        
        return loss
    
    def on_train_epoch_end(self):
        epoch_loss=np.average(self.train_loss)
        epoch_accuracy=np.average(self.train_acc)   ###logging train loss and accuracy in wandb
        
        
        self.train_loss.clear()
        self.train_acc.clear()
        
        
        #print(f"Epoch{self.current_epoch} Final train loss {epoch_loss} and final train accuracy {epoch_accuracy}")
        
    def validation_step(self,batch,batch_indx):
        img,label=batch
        img=img.float()
        
        ypred=self(img)
        
        loss=self.loss_fun(ypred,label)
         
        accuracy=(torch.argmax(ypred,dim=1)==label).sum()  ###calculating val loss and accuracy
        
        accuracy=accuracy/len(ypred)
        
        self.log('val_loss', loss,prog_bar=True)
        self.log('val_accuracy', accuracy,prog_bar=True)
        self.val_loss.append(loss.cpu().detach().numpy())
        self.val_acc.append(accuracy.cpu().detach().numpy())
        
        return loss
    
    def on_validation_epoch_end(self):
        epoch_loss=np.average(self.val_loss)
        epoch_accuracy=np.average(self.val_acc)
        
                                                     ###logging val loss and accuracy in wandb
        self.val_loss.clear()
        self.val_acc.clear()
        
        
        #print(f"Epoch{self.current_epoch} Final val loss {epoch_loss} and final val accuracy {epoch_accuracy}")
        
        
            
    def configure_optimizers(self):
        return [self.optimizer]


# In[4]:


def trainNN(epochs=20,data_aug=False,lr=1e-4):
    monitor=ModelCheckpoint(monitor='val_accuracy',save_top_k=1,mode='max')
    model=CNN_NATURALIST(data_aug=data_aug,lr=lr)
    trainer=pl.Trainer(accelerator='auto',max_epochs=epochs,callbacks=monitor)
    trainer.fit(model,train_loader,val_loader)


# In[ ]:


###command line arguments

parser = argparse.ArgumentParser()
 
parser.add_argument("-ep", "--epochs", default = 20, help = "Number of epochs ",type=int)
parser.add_argument('-aug','--augment',default=True,choices=[True,False],help='whether to do data augmentation or not')
parser.add_argument('-b','--batch_size',default=32,help='Number of batch_size for data loader',type=int)
parser.add_argument('-lr','--learning_rate',default=1e-4,help='learning rate of the model',type=float)
args = parser.parse_args()

###path for train and test####

train_path=r'D:\DL_DATA\naturalist\inaturalist_12K\train'
test_path=r'D:\DL_DATA\naturalist\inaturalist_12K\val'
transform=transforms.ToTensor()
###loading the train and test data as torch tensors###

train_data_=DatasetFolder(root=train_path,loader=load,extensions='.jpg',transform=transform)
test_data=DatasetFolder(root=test_path,loader=load,extensions='.jpg',transform=transform)

train_indx=[]
val_indx=[]

####taking the indices for train and val so that number of image in each class is same#####
for i in range(0,len(train_data_),1000):
    train_indx+=list(range(i,i+800))
    val_indx+=list(range(i+800,i+1000))
val_indx=val_indx[0:-1]


 #####splitting the 20% of train data into val data
train_data=Subset(dataset=train_data_,indices=train_indx)
val_data=Subset(dataset=train_data_,indices=val_indx)
###### creating data loader for train,val,test
train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,drop_last=True)
val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=True,drop_last=True)
test_loader=DataLoader(test_data,batch_size=args.batch_size,shuffle=True,drop_last=True)

 #### using albumentation library for augmentation

from torchvision import transforms
augmentor=transforms.Compose([
    transforms.RandomHorizontalFlip(p=.2),
    transforms.RandomVerticalFlip(p=.2),
    transforms.RandomAutocontrast(p=.2),
    transforms.RandomRotation(degrees=(-10,10)),
    transforms.RandomAdjustSharpness(sharpness_factor=0,p=.2),
    transforms.Resize((224,224))
])

parameter=trainNN(epochs=args.epochs,data_aug=args.augment,lr=args.learning_rate)

