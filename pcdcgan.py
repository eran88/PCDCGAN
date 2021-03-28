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
batch_size = 64
lr=0.0002
generator_scale=128   
discriminator_scale=128   
save_file=2
gen_steps=1
desc_steps=1
device = None
beta_gen=0.02
beta_disc=0.3

total_loss1=0
total_loss2=0




if torch.cuda.is_available():
    print("Using GPU.")
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    print("Using CPU.")
    device = torch.device("cpu")



 



def gan_loss(discriminator_value,discriminator_preds, labels,classes,beta,compute_loss2=True,):
    criterion = nn.BCELoss()
    loss = criterion(discriminator_value.squeeze(), labels)
    if(not compute_loss2):
        return loss
    criterion2 = nn.CrossEntropyLoss()

    loss_preds = criterion2(discriminator_preds, classes)
    global total_loss1,total_loss2
    total_loss1=total_loss1+loss.item()
    total_loss2=total_loss2+loss_preds.item()
    return loss+beta*loss_preds 

def train_cgan(generator, discriminator, train_loader, lr=0.0001, latent_vector_size=100, nepochs=100, print_freq=400, device="cuda",batch_size=32,file_name="models",probs=False,losses=[]):
    '''
    The function trains a gan model.
    '''
    # Create optimizers for the discriminator and generator
    #   each optimizes different model
    global total_loss1,total_loss2,beta_gen,beta_disc
    d_optimizer = optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))

    # start train loop
    mydesc_step=0
    global desc_steps
    for epoch in range(nepochs):
        # set both networks to train mode
        discriminator.train()
        generator.train()

        for batch_i, (real_images, real_classes) in enumerate(train_loader):
            # rescale images before training
            real_images = (real_images*2 - 1).to(device)
            # get batch size
            batch_size = real_images.size(0)
            real_classes = real_classes.to(device)
            ######### A. TRAIN THE DISCRIMINATOR #########
            d_optimizer.zero_grad()
            mydesc_step=mydesc_step+1
                # 1. Compute the discriminator loss on real images
            discriminator_batch_labels = create_labels(batch_size, device, real_data=True, for_discriminator=True)
            disc_output,disc_preds=discriminator(real_images)
            loss1 = gan_loss(disc_output,disc_preds, discriminator_batch_labels,real_classes,beta_disc)

            # 2. Generate fake images using the generator (use ```create_latent_batch_vectors```)
            latent_batch,classes = create_latent_batch_vectors(batch_size=batch_size, latent_vector_size=latent_vector_size, device=device,class_size=class_size,probs=probs)
            generator_images = generator(latent_batch,classes)

            # 3. Compute the discriminator loss on fake images
            generator_batch_labels = create_labels(batch_size, device, real_data=False, for_discriminator=True)
            disc_output,disc_preds=discriminator(generator_images)
            loss2 = gan_loss(disc_output,disc_preds, generator_batch_labels,classes.long(),beta_disc,compute_loss2=False)

            # 4. Calculate discriminator total loss and do backprop
            discriminator_loss = loss1 + loss2
            discriminator_loss.backward()
            ######################
            d_optimizer.step()
            if mydesc_step%desc_steps!=0:
                continue
            ######### B. TRAIN THE GENERATOR #########
            for i in range(gen_steps):
                g_optimizer.zero_grad()

                # 1. Generate fake images
                latent_batch,classes = create_latent_batch_vectors(batch_size=batch_size, latent_vector_size=latent_vector_size, device=device,class_size=class_size,probs=probs)
                generator_images = generator(latent_batch,classes)

                # 2. Compute the discriminator loss on them but with flipped labels.
                flipped_labels = create_labels(batch_size, device, real_data=False, for_discriminator=False)
                disc_output,disc_preds=discriminator(generator_images)
                generator_loss = gan_loss(disc_output,disc_preds, flipped_labels,classes.long(),beta_gen)

                # 3. perform backprop
                generator_loss.backward()
                ######################
                g_optimizer.step()
        print(f'Epoch {epoch}, Batch {batch_i}, Disc_loss: {discriminator_loss.item()}, Gen_loss: {generator_loss.item()}, total_loss1: {int(total_loss1)}, total_loss2: {int(total_loss2)}')
        total_loss1=0
        total_loss2=0
        if((epoch+1)%save_file==0):
            with open(file_name, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([generator, discriminator,losses], f)
                print("saved file to: "+file_name)
        # keep track of losses
        losses.append((discriminator_loss.item(), generator_loss.item()))

    generator.eval()
    return losses




def main():

    file_name="models"
    if(len(sys.argv)>1):
        file_name=sys.argv[1]
    file_name=file_name+".pkl"  
    print(file_name)
    my_file = Path(file_name)

    if my_file.is_file() and len(sys.argv)>1:
        with open(file_name,"rb") as f:  
            gen, disc,losses= pickle.load(f)
            gen.to(device)
            disc.to(device)
            if(len(sys.argv)>2 and sys.argv[2]=="create"):
                return create(gen)

    # file exists
    else:
        print("new file")
        gen=Generator(100,generator_scale).to(device)
        disc=Discriminator_new(discriminator_scale).to(device)
        gen.apply(weights_init)
        disc.apply(weights_init)
        losses=[]
 
    mydata=load_data("list_patition_label.txt")
    classess_options=['Surprise',"Fear","Disgust","Happiness","Sadness","Anger","Neutral"]
    train_tfms = transforms.Compose([transforms.ToTensor()])
    my_dataset = CustomDataSet("aligned/",mydata,classess_options, transform=train_tfms)
    train_loader = DataLoader(my_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    print(my_dataset.sizes)
    total=my_dataset.total_size
    sizes=my_dataset.sizes
    probs=[]
    for i in range(len(sizes)):
        probs.append(sizes[i]/total)      
    print(probs)
    losess=train_cgan(gen, disc, train_loader, lr=lr, latent_vector_size=latent_vector_size, nepochs=300, print_freq=5, device=device,file_name=file_name,probs=probs,losses=losses)

if __name__ == '__main__':
    main()
