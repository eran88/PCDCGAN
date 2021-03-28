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





def main():

    if(len(sys.argv)>1):
        file_name=sys.argv[1]
    else:
        raise Exception ("You must enter a file name ")
    file_name=file_name+".pkl"  
    print(file_name)
    my_file = Path(file_name)

    if not my_file.is_file() or len(sys.argv)<=1:
        raise Exception ("file not exist")
    gen = None
    with open(my_file, 'rb') as f:
        gen, _, _ = pickle.load(f)
    return create(gen)
if __name__ == '__main__':
    main()
