from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device("cpu")
import pickle
latent_vector_size=100
class_size=7
def load_data(filename):
    ans={}
    with open(filename) as f: 
        lines = [line.rstrip('\n') for line in f]
        for line in lines:
            words=line.split()
            if(len(words)>1):
                ans[words[0]]=int(words[1])-1
    return ans
class CustomDataSet(Dataset):
    def __init__(self, main_dir,data,classess_options, transform,only_label=False,type=False):
        self.transform = transform
        self.total_imgs = os.listdir(main_dir)
        self.labels=[]
        self.images=[]
        self.path=main_dir
        self.sizes=[]
        self.total_size=0
        self.classess_options=classess_options
        imgs = os.listdir(main_dir)
        for i in range(len(classess_options)):
            self.sizes.append(0)
        for i in range(len(imgs)):
            img=imgs[i]
            img_path=main_dir+img   
            if img in data:
                label=data[img]
                if not only_label or only_label==label:
                    if not type or type in img:
                        self.images.append(img_path)
                        self.labels.append(label)
                        self.total_size=self.total_size+1
                        self.sizes[label]=self.sizes[label]+1
            

    def __len__(self):
        return self.total_size


    def __getitem__(self, idx):               
        image = Image.open(self.images[idx])
        tensor_image = self.transform(image)
        return [tensor_image.to(device),self.labels[idx]]

matrix = torch.zeros((class_size, class_size, 48, 48))
for i in range(class_size):
    matrix[i, i, :, :] = 1

def make_matrix_onehot(classes):
    ans=[]
    for i in range(len(classes)):
        ans.append(matrix[classes[i]])
    ans=torch.stack(ans).to(device)
    return ans

def one_hot(num, index):
    a = torch.zeros(num)
    a[index] = 1
def one_hot_classes(classes):
    a=torch.zeros(len(classes),class_size)
    for i in range(len(classes)):
        a[i][classes[i]] = 1
    return a.to(device)
def random_classes(num, batch_size,probs):
    indexes =np.random.choice(num, batch_size, p=probs,replace=True)
    return indexes
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def create_latent_batch_vectors(batch_size, latent_vector_size,class_size, device,probs=False):
    '''
    The function creates a random batch of latent vectors with random values 
    distributed uniformly between -1 and 1. 
    Finally, it moves the tensor to the given ```device``` (cpu or gpu).
    The output should have a shape of [batch_size, latent_vector_size].
    '''
    z = torch.Tensor(np.random.uniform(-1, 1, size=(batch_size, latent_vector_size)))
    c=random_classes(class_size, batch_size,probs)
    return [z.to(device),torch.Tensor(c).to(device)]
def create_labels(batch_size, device, real_data=True, for_discriminator=True):
  '''
  The function returns a batch_size-tensor with the correct labels.
  It creates labels according to the ```real_data``` and ```for_discriminator``` parameters.
  ```real_data``` - True if the batch images are real, and False for fake images
  ```for_discriminator``` - True if the labels should be created for the discriminator loss,
                            and False for the generator loss (False)
  If ```for_discriminator``` is set to True, it returns labels with smoothing,
  otherwise, it creates labels for the generator without smoothing.
  Finally, it moves the label tensors to the specified ```device```
  '''
  res = None

  if real_data == True:
    res = torch.ones(batch_size)-0.1
  else:
    if for_discriminator:
      res = torch.zeros(batch_size)
    else:
      res = torch.ones(batch_size)

  return res.to(device)
def show_images(imgs,title="Training Images",size=32,figsize=(7,7)):
    print(imgs.shape)
    #imgs=torch.reshape(imgs,(imgs.shape[0],1,48,48))
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(imgs[:size+1], padding=2, normalize=True,nrow =7 ).cpu(),(1,2,0)))
    plt.show()
def create(gen):
    gen.eval()
    noise,classes=create_latent_batch_vectors(49, latent_vector_size,class_size, device,probs=[1/7,1/7,1/7,1/7,1/7,1/7,1/7])
    b=gen(noise,classes).detach()
    show_images(b,"generator images",size=49,figsize=(7,7))
    noise,classes=create_latent_batch_vectors(7, latent_vector_size,class_size, device,probs=[1/7,1/7,1/7,1/7,1/7,1/7,1/7])
    news=[]
    classes=[]
    for i in range(7):
        for j in range(class_size):
            news.append(noise[i])
            classes.append(j)
    classes=torch.Tensor(classes).to(device)
    noise = torch.stack(news).to(device)
    b=gen(noise,classes).detach()
    show_images(b,"Surprise  |  Fear  |  Disgust  |  Happiness  |  Sadness  |  Anger  |  Neutral",size=48,figsize=(8,8))


    return

def file_save(file_name,generator, discriminator,losses,device):
    with open(file_name, 'wb') as f:
        cpu_device=torch.device("cpu")
        generator.to(cpu_device)
        discriminator.to(cpu_device)    
        pickle.dump([generator, discriminator,losses], f)
        generator.to(device)
        discriminator.to(device)        
        print("saved file to: "+file_name)    