# deep-anomaly-classification

This repository contains an implementation of the paper [Deep One-Class Classification](http://proceedings.mlr.press/v80/ruff18a.html), an image anomaly classification method. 

The original paper proposes its method to find anomalies on CIFAR10 and MNIST datasets, the authors also benchmark it on adversarial attacks. Here  I extend their work it to higher resolution datasets, like 128x128.

Original Repo: https://github.com/lukasruff/Deep-SVDD-PyTorch

# Deep SVD

In this repo, I'll use the Deep SVDD is an anomaly classification for golden sample datasets. 

So if you have something that you need to classify when anomalies happen but you don't know the anomalies (i.e you only have good samples, bad samples rarely occur), this is a good Unsupervised setup and might suit you!

## Overview

**Deep SVDD**: Learn a function that maps input space to a smaller vector space, classify if an input is anomalous or not in that smaller space.


**Objetive** : Train a model to the best representation of a dataset in a smaller vector space, then optimize a hypersphere that contains the samples of good data inside of it. At test time, everything outside the hypersphere will be considered an anomaly.

![image](https://user-images.githubusercontent.com/56324869/178617483-5068461d-7fe5-41ca-a6a2-3122de27f626.png)

When we finish, we'll have a model, a hypersphere center C and its radius R in F vector space.

# Training
## 1º Step

We train a Deep Auto Encoder to reconstruct our dataset.

## 2º Step
We take only the Encoder part of our AutoEncoder, and its last convolutional layer as a latent space, which contains the smallest representation of our data (512 dimensions in my model).

## 3º Step
We create a 512d vector and calculate the mean of our dataset at the latent dimension (last conv layer). We use it as the center of our 512d hypersphere.

## 4º Step
We optimize the hypersphere radius in such a way that in the end we have a 512d hypersphere that contains our good samples inside of it.

## 5º Step
We simply forward new images through our encoder, it outputs a 512d vector (the image in the latent space), then we simply check whether that 512d point is inside the hypersphere or not.

Scores smaller than 0 are considered inside of our hypersphere, whereas scores higher than 0 are anomalies.

## Results:

I've trained the golden sample image with 199 augmentations (noise, contrasts, and other augmentations that dont change the core of our class).

Ideally, we should be able to get negative scores for our good samples, but that doesnt happen everytime, instead, we have really small values as the output, like 0.0x or 0.x, which deppends on the training (I've got it right sometimes, it deppends on training and hyperparameters, and specially, on data).

### Golden sample:

![89](https://user-images.githubusercontent.com/56324869/178614325-abac2fcf-e48e-4544-a330-58ec3677a4aa.jpg)

Score:  0.01209

### Darker

![89 jpg-augmentation-20](https://user-images.githubusercontent.com/56324869/178614337-a6621e99-14b2-4b1f-b442-c40b69135176.jpg)

Score: 0.11798178584807317

### Handmade defects:

![f8-anomalous](https://user-images.githubusercontent.com/56324869/178614356-3b674bd6-43d4-4e4f-89ab-593e55494c08.jpg)

Score: 32.154

### Shifted painting

![f8-moved](https://user-images.githubusercontent.com/56324869/178614360-a0cfafed-a23a-49c4-9f55-cec9a2906f93.jpg)

Score: 159.10847383172745

### F12 instead of F8 written

![f8-small-defect](https://user-images.githubusercontent.com/56324869/178614361-e9be41e2-a95f-44de-bcc4-7ab2e43aa804.jpg)

Score: 2.3384528699659626

## Citation

I'am a research engineer, so I don't really write papers on the subject, instead I write the models and make them work on real world scenarios. 
If you find that repository useful, please make a reference to it!