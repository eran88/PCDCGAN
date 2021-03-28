# Introduction
In this project, we compared the traditional CDCGAN (Conditional Deep Convolutional Generative adversarial network) with PCDCGAN (Predictive Deep Convolutional Generative Adversarial Network)

## How to start
Firstable clone the repository to your computer, then open the command line on the path of the cloned repository.

## To use pretrained models
1. Download one of the models from the following link: (These files are the trained weights of the models etc.) 
https://drive.google.com/drive/folders/1PWxUckhEZdz7VetuqyQUffeUO1SWzh4i?usp=sharing

2. To create images from the models type in the command line:
`python create.py "file_name_without_pkl"`

3. To evaluate the model with Inception Score and Frechet Inception Distance type in the command line:
`python evalute.py "file_name_without_pkl"`

## Train and use a new models
To train and use a new pcdcgan model type in the command line:
`python pcdcgan.py "file_name_without_pkl"`

To train and use a new cdcgan model type in the command line:
`python cdcgan.py "file_name_without_pkl"`

You can keep training the new models/pretrained models with the same command.
Enjoy!
