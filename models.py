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
class_size=7
from funcs import *

class Generator(nn.Module):
    def __init__(self, input_size,scale):
        super(Generator, self).__init__()
        self.emb=torch.nn.Embedding(num_embeddings=class_size, embedding_dim=7)
        self.conv1=nn.ConvTranspose2d( input_size+class_size, scale*12, kernel_size=4, stride=1, padding=0, bias=False)
        self.conv2=nn.ConvTranspose2d( scale*12, scale*8, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv3=nn.ConvTranspose2d( scale*8, scale*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4=nn.ConvTranspose2d( scale*4, scale*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5=nn.ConvTranspose2d( scale*2, scale, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv6=nn.ConvTranspose2d( scale, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1=nn.BatchNorm2d(scale*12)
        self.bn2=nn.BatchNorm2d(scale*8)
        self.bn3=nn.BatchNorm2d(scale*4)
        self.bn4=nn.BatchNorm2d(scale*2)
        self.bn5=nn.BatchNorm2d(scale)
    def forward(self, x,classes):

        classes=classes.to(torch.int64)
        batch_size=x.shape[0]
        emb=one_hot_classes(classes)
        x=torch.cat((x,emb),1)
        x=torch.reshape(x, (batch_size,x.shape[1],1,1))
        x=F.relu(self.bn1(self.conv1(x)))     
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=F.relu(self.bn4(self.conv4(x)))   
        x=F.relu(self.bn5(self.conv5(x)))
        output=torch.tanh(self.conv6(x))
        return output
class Discriminator_new(nn.Module):
    def __init__(self,scale):
        super(Discriminator_new, self).__init__()
        self.conv1=nn.Conv2d( 3, scale, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2=nn.Conv2d( scale, scale*2, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv3=nn.Conv2d( scale*2, scale*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4=nn.Conv2d( scale*4, scale*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5=nn.Conv2d( scale*8, scale*12, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv6=nn.Conv2d( scale*12, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1=nn.BatchNorm2d(scale)
        self.bn2=nn.BatchNorm2d(scale*2)
        self.bn3=nn.BatchNorm2d(scale*4)
        self.bn4=nn.BatchNorm2d(scale*8)
        self.bn5=nn.BatchNorm2d(scale*12)
        self.preds_fc2=nn.Linear(scale*12*16, class_size)
        self.scale=scale
    def forward(self, x):
        batch_size=x.shape[0]      
        x=F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)                
        x=F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x=F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x=F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x=F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        output=self.conv6(x)
        output=torch.reshape(output, (batch_size,1))
        preds=torch.reshape(x, (batch_size,16*self.scale*12))
        preds=self.preds_fc2(preds)
        output=torch.sigmoid(output)
        return [output,preds]



class Discriminator_cgan(nn.Module):
    def __init__(self,scale):
        super(Discriminator_cgan, self).__init__()
        self.conv1=nn.Conv2d( 4, scale, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2=nn.Conv2d( scale, scale*2, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv3=nn.Conv2d( scale*2, scale*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4=nn.Conv2d( scale*4, scale*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5=nn.Conv2d( scale*8, scale*12, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv6=nn.Conv2d( scale*12, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1=nn.BatchNorm2d(scale)
        self.bn1_emb=nn.BatchNorm1d(100*100)
        self.bn2=nn.BatchNorm2d(scale*2)
        self.bn3=nn.BatchNorm2d(scale*4)
        self.bn4=nn.BatchNorm2d(scale*8)
        self.bn5=nn.BatchNorm2d(scale*12)
        self.scale=scale
        self.emb_fc = nn.Linear(class_size, 100 * 100)
        self.emb=torch.nn.Embedding(num_embeddings=class_size, embedding_dim=7)
    def forward(self, x,classes):
        batch_size=x.shape[0]
        emb=self.emb(classes)
        emb=F.leaky_relu(self.bn1_emb(self.emb_fc(emb)), negative_slope=0.2)
        emb=torch.reshape(emb, (batch_size,1,100,100))
        x=torch.cat((x,emb),1)
        x=F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        
        
        x=F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x=F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        
        x=F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x=F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        output=self.conv6(x)
        output=torch.reshape(output, (batch_size,1))
        output=torch.sigmoid(output)
        return output
     
        
        