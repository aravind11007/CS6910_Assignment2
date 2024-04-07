# CS6910_Assignment2 Part A
The goal of this assignment is twofold: (i) train a CNN model from scratch and learn how to tune the hyperparameters and visualize filters
## Problem Statement
Build a small CNN model consisting of 5 convolution layers. Each convolution layer would be followed by an activation and a max-pooling layer.After 5 such 
conv-activation-maxpool blocks, you should have one dense layer followed by the output layer containing 10 neurons (1 for each of the 10 classes). 
The input layer should be compatible with the images in the iNaturalist dataset dataset.
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
* A python class inherented from torch.nn is created for implementing the CNN model. The model takes in the number of filter, filter size,
  whether to do batch_norm or not ,the drop_out probability, activation funtion and number of neuron in the hidden layer as the argument
```
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
```
* For training and validating the model, I have used pytorch-lightning class and was named it as CNN_NATURALIST, it takes in the number of filter, filter size,organisation
  whether to do batch_norm or not, the drop_out probability, activation funtion and number of neuron in the hidden layer,the optimizer as the argument
* This class CNN_NATURALIST was included in a another function called Train_NN for wandb sweep
```
def train_NN(num_filter=64,filter_size=3,organisation='same',data_aug=False,batch_norm=False,drop_out=.2,
                act='ReLU',log=False,epochs=20,lr=1e-4,neuron=128,optimizer='sgd'):
    model=CNN_NATURALIST(num_filter=num_filter,filter_size=filter_size,organisation=organisation,
                         data_aug=data_aug,batch_norm=batch_norm,drop_out=drop_out,act=act,log=log,lr=lr,neuron=neuron,
                        optimizer=optimizer)
    trainer=pl.Trainer(accelerator='auto',max_epochs=epochs)
    trainer.fit(model,train_loader,val_loader)

```
* Following where the different hyperparameter values
```
  parameters_dict={
    'num_filter':{
        'values':[32,64]
    },
    'filter_size':{
        'values':[3,5]
    },
    'organisation':{
        'values':['same','doubling','halving']
    },
    'data_aug':{
        'values':[True,False]
    },
    'batch_norm':{
      'values':[True,False]  
    },
    'drop_out':{
        'values':[.2,.3]
    },
    'act':{
        'values':['ReLU','GELU','SiLU','Mish']
    },
    'epochs':{
        'values':[10,15,20]
    },
    'lr':{
        'values':[1e-3,1e-4,1e-5]
    },
    'neuron':{
        'values':[128,256,512]
    },
    'optimizer':{
        'values':['sgd','adam']
    }
}
```
## Code specifications
A python script train_partA.py was created that accepts the following command line arguments with the specified values -
```
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-a`, `--activation` | 'SiLU' | choices:  ['ReLU','GELU','SiLU','Mish'] |
| `-e`, `--epochs` | 20 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-f`, `--filter_size` | 3 | choices:  [3,5] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "adam"] | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `-bn`, `--batch_norm` | True | choices=[True,False] |
| `-aug`, `--augment` | True | choices=[True,False] | 
| `-d`, `--drop_out` | 0.3 | choices=[0.2,0.3] | 
| `-o`, `--organisation` | doubling | choices=['same','doubling','halving'] |
| `-lg`, `--logs` | False | choices = ["True","False"] |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-neu`, `--neuron` | 512 | "Number of neurons in hidden layer" |
```
