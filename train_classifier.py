import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import sys
import pickle
from collections import defaultdict
from funcs import *
from models import *
from pathlib import Path
import torchvision.utils as vutils
latent_vector_size = 100
class_size=7
batch_size = 16
lr=0.0002
scale=128   
save_file=2
torch.cuda.empty_cache()


if torch.cuda.is_available():
    print("Using GPU.")
    device = torch.device('cuda:0')
else:
    print("Using CPU.")
    device = torch.device("cpu")



 





def freeze_network(net):
  '''
  The function freezes all layers of a pytorch network (net).
  '''

  for name, param in net.named_parameters():
    if(name!="fc.weight" and name!="fc.bias"):
        param.requires_grad = False
    else:
        print("dont change here")
def train( train_loader,val_loader, lr=0.00001, device="cuda",file_name="models"):

    import torch
    model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True,aux_logits=False)  
    model.train()       
    model.fc = nn.Linear(2048, 7).to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)  
    for epoch in range(20):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

        print("Epoch "+str(epoch))
        print_accuracy(model, train_loader, batch_size, "Train Set")
        print_accuracy(model, val_loader, batch_size, "Validation Set")
        torch.save(model, "classifier.txt")

def print_accuracy(model, data, batch_size, data_name):
    good = 0

    model.eval()
    for x, y in data:
        y=y.to(device)
        outputs = model(x)
        max_arg_outputs = torch.max(outputs, 1)[1]
        good += max_arg_outputs.eq(y.data.view_as(max_arg_outputs)).sum()

    size = len(data)* batch_size
    accuracy = 100 * good/size
    print("{}:  Accuracy: {}/{} ({:.0f}%)".format(data_name, good, size , accuracy))

def main():

    file_name="models"
    if(len(sys.argv)>1):
        file_name=sys.argv[1]
    file_name=file_name+".pkl"  
    print(file_name)
    my_file = Path(file_name)


 
    mydata=load_data("list_patition_label.txt")
    classess_options=['Surprise',"Fear","Disgust","Happiness","Sadness","Anger","Neutral"]
    train_tfms = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    
    my_dataset = CustomDataSet("aligned/",mydata,classess_options, transform=train_tfms)

    train_set = CustomDataSet("aligned/",mydata,classess_options, transform=train_tfms,type="train")
    val_set = CustomDataSet("aligned/",mydata,classess_options, transform=train_tfms,type="test")
    #train_set, val_set = torch.utils.data.random_split(my_dataset, [int(len(my_dataset)*0.85), len(my_dataset)-int(len(my_dataset)*0.85)])
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True) 
    train( train_loader,val_loader, device=device,file_name=file_name)

if __name__ == '__main__':
    main()
