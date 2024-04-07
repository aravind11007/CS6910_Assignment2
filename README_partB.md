# CS6910_Assignment2 Part B
The goal of this assignment is to Fine-tuning a pre-trained model
## Problem Statement
In most DL applications, instead of training a model from scratch, you would use a model pre-trained on a similar/related task/dataset. From torchvision, you can load ANY ONE model (GoogLeNet, InceptionV3, ResNet50, VGG, EfficientNetV2, VisionTransformer etc.) pre-trained on the ImageNet dataset. Given that ImageNet also contains many animal images, it stands to reason that using a model pre-trained on ImageNet maybe helpful for this task.You will load a pre-trained model and then fine-tune it using the naturalist data that you used in the previous question
## Process
* The data was download from the link
* The data was loaded as pytorch tensor using "DatasetFolder" library of torchvision
```
train_data_=DatasetFolder(root=train_path,loader=load,extensions='.jpg',transform=transforms)
test_data=DatasetFolder(root=test_path,loader=load,extensions='.jpg',transform=transforms)

```
* The val data was created from 20% of train data
* The train, val and test data loader was created using the torch dataloader
```
train_loader=DataLoader(train_data,batch_size=64,shuffle=True,drop_last=True)
val_loader=DataLoader(val_data,batch_size=64,shuffle=True,drop_last=True)
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,drop_last=True)
```
*For training and validating the model, I have used pytorch-lightning class and was named it as CNN_NATURALIST. The pretrained
model VGG19 is loaded inside this class
```
class CNN_NATURALIST(pl.LightningModule):
    def __init__(self,data_aug=False):
        super().__init__()
        self.data_aug=data_aug
        
        
        model=torchvision.models.vgg19(pretrained=True)
        for params in model.features[0:15].parameters():
            params.requires_grad=False
        self.model=model
        self.model.classifier[6]=nn.Linear(4096,10)
        
        self.loss_fun=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4)
        
        self.train_loss=[]
        self.train_acc=[]
        self.val_loss=[]
        self.val_acc=[]
        
    def forward(self,x):
        
        ypred=self.model(x)
        #ypred=nn.Softmax(dim=1)(ypred)
            
        return ypred
    
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
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss.append(loss.cpu().detach().numpy())
        self.train_acc.append(accuracy.cpu().detach().numpy())
        
        return loss
    
    def on_train_epoch_end(self):
        epoch_loss=np.average(self.train_loss)
        epoch_accuracy=np.average(self.train_acc)
        
        
        self.train_loss.clear()
        self.train_acc.clear()
        
        wandb.log({"Train_Accuracy":np.round(epoch_accuracy*100,2),"Train_Loss":epoch_loss,"Epoch":self.current_epoch})
        
        
        #print(f"Epoch{self.current_epoch} Final train loss {epoch_loss} and final train accuracy {epoch_accuracy}")
        
    def validation_step(self,batch,batch_indx):
        img,label=batch
        img=img.float()
        
        ypred=self(img)
        
        loss=self.loss_fun(ypred,label)
        
        accuracy=(torch.argmax(ypred,dim=1)==label).sum()
        
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
        self.val_acc.clear()
        
        wandb.log({"Val_Accuracy":np.round(epoch_accuracy*100,2),"Val_Loss":epoch_loss,"Epoch":self.current_epoch})
        
        
        #print(f"Epoch{self.current_epoch} Final val loss {epoch_loss} and final val accuracy {epoch_accuracy}")
        
        
            
    def configure_optimizers(self):
        return [self.optimizer]

```
* This pytorch lightning class is added in an another function called trainNN
```
def trainNN(epochs=20,data_aug=False,lr=1e-4):
    monitor=ModelCheckpoint(monitor='val_accuracy',save_top_k=1,mode='max')
    model=CNN_NATURALIST(data_aug=data_aug,lr=lr)
    trainer=pl.Trainer(accelerator='auto',max_epochs=epochs,callbacks=monitor)
    trainer.fit(model,train_loader,val_loader)

```
## Code specifications
A python script train_partB.py was created that accepts the following command line arguments with the specified values -
```
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-e`, `--epochs` | 20 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `-aug`, `--augment` | True | choices=[True,False] | 
