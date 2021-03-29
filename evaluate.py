from numpy import log
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from numpy import cov
from numpy import expand_dims
from numpy import asarray
from numpy import mean
from numpy import exp
import torchvision.utils as vutils
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

latent_vector_size = 100
class_size = 7
batch_size = 20
lr = 0.0002
generator_scale = 128
discriminator_scale = 128

device = None

torch.no_grad()


if torch.cuda.is_available():
    print("Using GPU.")
    device = torch.device('cuda:0')
else:
    print("Using CPU.")
    device = torch.device("cpu")


def creates_images_by_class(generator_model, emotion_label):
    generator_model.eval()
    noise_images, classes = create_latent_batch_vectors(
        batch_size, latent_vector_size, class_size, device, probs=[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
    if emotion_label != -1:
        classes = np.zeros(batch_size)+emotion_label
        classes = torch.from_numpy(classes).long().to(device)
    return generator_model(noise_images, classes)


# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score


def create_activations_real(loader, classifier, train_tfms, softmax, rounds):
    preds_all = False
    First = True
    for images, _ in loader:
        images = train_tfms(images)
        preds = classifier(images)
        preds = softmax(preds)
        preds = asarray(preds.detach().cpu())
        rounds = rounds-1
        if not First:
            preds_all = np.concatenate((preds_all, preds), axis=0)
        else:
            First = False
            preds_all = preds
        if rounds == 0:
            return preds_all
    return preds_all


def create_activations(gen, classifier, train_tfms, softmax, rounds):
    preds_all = False
    for i in range(rounds):
        images = creates_images_by_class(gen, -1)
        images = train_tfms(images)
        preds = classifier(images)
        preds = softmax(preds)
        preds = asarray(preds.detach().cpu())
        if i > 0:
            preds_all = np.concatenate((preds_all, preds), axis=0)
        else:
            preds_all = preds
    return preds_all


def Identity(x):
    return x


def main():
    if(len(sys.argv) > 1):
        file_name = sys.argv[1]
    else:
        raise Exception("You must enter a file name ")

    file_name = file_name+".pkl"
    print(file_name)
    my_file = Path(file_name)

    if not my_file.is_file() or len(sys.argv) <= 1:
        raise Exception("file not exist")
    gen = None
    with open(my_file, 'rb') as f:
        gen, _, _ = pickle.load(f)
    mydata = load_data("list_patition_label.txt")
    classess_options = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

    gen.to(device)
    classifier = torch.load("classifier.txt", map_location=device).to(device)
    classifier.eval()
    classifier.to(device)
    gen.eval()
    train_tfms = transforms.Compose([transforms.Resize(299), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    dataset_transform = transforms.Compose([transforms.ToTensor(),])
    my_dataset = CustomDataSet("aligned/", mydata, classess_options, transform=dataset_transform)
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    softmax = torch.nn.Softmax(dim=1)

    all_score = 0
    for j in range(10):
        preds_all = create_activations(gen, classifier, train_tfms, softmax, 250)
        score = calculate_inception_score(preds_all)
        print(f'The inception score test {j}: {score}')
        all_score = all_score+score
    print(f'The inception score average: {all_score/10}')
    all_score = 0
    classifier.fc = torch.nn.Identity()

    #uncomment for testing frechet distance on the pretrained model
    #classifier = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True,aux_logits=False).to(device)
    for j in range(20):
        preds_images = create_activations_real(train_loader, classifier, train_tfms, Identity, 150)
        preds_all = create_activations(gen, classifier, train_tfms, Identity, 150)
        #print(preds_all.shape)
        score = calculate_fid(preds_all, preds_images)
        print(f'frechet inception distance test {j}: {score}')
        all_score = all_score+score
    print(f'The frechet inception distance: {all_score/20}')

    ##calcuating Frechet Inception Distance
if __name__ == '__main__':
    main()