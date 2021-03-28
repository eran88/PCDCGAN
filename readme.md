# Introduction
In this project, we compared the traditional CDCGAN (Conditional Deep Convolutional Generative adversarial network) with PCDCGAN (Predictive Deep Convolutional Generative Adversarial Network)

<br />

## How to start
Firstable clone the repository to your computer, then open the command line on the path of the cloned repository.

<br />

## To use pretrained models
1. Download one of the models from the following link:
https://drive.google.com/drive/folders/1PWxUckhEZdz7VetuqyQUffeUO1SWzh4i?usp=sharing <br />
This files contains the trained weights of the models etc. After you've downloaded the file place it inside the root of the repository folder.

2. To create images from the models type in the command line:
`python create.py "file_name_without_pkl"`

3. To evaluate the model with Inception Score and Frechet Inception Distance type in the command line:
`python evalute.py "file_name_without_pkl"`

<br />

## Train and use a new models
To train and use a new pcdcgan model type in the command line:
`python pcdcgan.py "file_name_without_pkl"`

To train and use a new cdcgan model type in the command line:
`python cdcgan.py "file_name_without_pkl"`

You can keep training the new models/pretrained models with the same command.
Enjoy!
