In this project we compared th traditional cdcgan (Conditional Deep Convolutional Generative adversarial network) with PCDCGAN (Predictive Deep Convolutional Generative Adversarial Network)
# To use pretrained images:
1. download at least one of the models from:
https://drive.google.com/drive/folders/1PWxUckhEZdz7VetuqyQUffeUO1SWzh4i?usp=sharing
2. To create images from the models type:
python create.py "file_name_without_pkl"
3. To evalute the model with Inception Score and Frechet Inception  Distance type:
python evalute.py "file_name_without_pkl"

# To train and use a new pcdcgan model type:
1. python pcdcgan.py "file_name_without_pkl"
To train and use a new cdcgan model type:
1. python cdcgan.py "file_name_without_pkl"

You can keep training the new models/pretrained models with the same commands
