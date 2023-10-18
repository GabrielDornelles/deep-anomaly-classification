# deep-anomaly-classification

This repository contains an implementation of the paper [Deep One-Class Classification](http://proceedings.mlr.press/v80/ruff18a.html), an image anomaly classification method. 

The original paper proposes its method to find anomalies on CIFAR10 and MNIST datasets, the authors also benchmark it on adversarial attacks. Here, I extend their work to higher resolution datasets, like 128x128 pixels.

Original Repo and implementation: https://github.com/lukasruff/Deep-SVDD-PyTorch

# Deep SVDD

In this repo, I'll use the Deep SVDD as an anomaly classification for golden sample datasets. 

So, if you have something that you need to classify when anomalies happen but you don't know the anomalies (i.e you only have good samples, bad samples rarely occur), this is a good Unsupervised setup and might suit you!

# Deep Support Vector Data Description (Deep SVDD)

Deep SVDD is a method that utilizes deep convolutional neural networks to learn a function that maps the input space into a smaller vector space. The goal is to make non-anomalous points close to each other in that space. The convolutional neural network learns to map these points close to each other based on the most important characteristics of the class.

![Deep SVDD](https://user-images.githubusercontent.com/56324869/178617483-5068461d-7fe5-41ca-a6a2-3122de27f626.png)

## Objective

The main objective of Deep SVDD is to train a model to find the best representation of a dataset in a smaller vector space. It then optimizes a hypersphere that contains the samples of good data inside it. During testing, everything outside the hypersphere is considered an anomaly.

## Training

### Step 1: Deep Auto Encoder Training

In the first step, we train a Deep Auto Encoder to reconstruct our dataset. A good reconstruction implies that the encoder has learned good features, and the bottleneck has a strong representation of the trained class.

### Step 2: Latent Space Extraction

In the second step, we take only the encoder part of our AutoEncoder and its last convolutional layer as a latent space, which contains the smallest representation of our data (typically 512 dimensions).

### Step 3: Hypersphere Center Calculation

We create a 512-dimensional vector and calculate the mean of our dataset at the latent dimension (last conv layer). This mean serves as the center of our 512-dimensional hypersphere.

### Step 4: Hypersphere Radius Optimization

In this step, we optimize the hypersphere radius to ensure it contains our good samples inside it.

### Step 5: Anomaly Detection

To detect anomalies, we forward new images through our encoder, which outputs a 512-dimensional vector (the image in the latent space). We check whether this 512-dimensional point is inside the hypersphere or not. Scores smaller than 0 are considered inside the hypersphere, while scores higher than 0 are anomalies.

## Results

The results can vary based on training and data, and ideal outcomes should involve negative scores for good samples. However, it's common to have very small values as the output. Also, Its recommended to apply some augmentations to help the model not overfit your data (it will happen if you train with 1 or 2 images).

### Example Results:

| Image | Score |
|-------|-------|
| ![Golden Sample](https://user-images.githubusercontent.com/56324869/178614325-abac2fcf-e48e-4544-a330-58ec3677a4aa.jpg) | 0.012 |
| ![Darker](https://user-images.githubusercontent.com/56324869/178614337-a6621e99-14b2-4b1f-b442-c40b69135176.jpg) | 0.11 |
| ![Handmade Defects](https://user-images.githubusercontent.com/56324869/178614356-3b674bd6-43d4-4e4f-89ab-593e55494c08.jpg) | 32.15 |
| ![Shifted Painting](https://user-images.githubusercontent.com/56324869/178614360-a0cfafed-a23a-49c4-9f55-cec9a2906f93.jpg) | 159.10 |
| ![F12 Instead of F8](https://user-images.githubusercontent.com/56324869/178614361-e9be41e2-a95f-44de-bcc4-7ab2e43aa804.jpg) | 2.33 |


## Citation

I'am a research engineer, so I don't really write papers on the subject, instead I write the models and make them work on real world scenarios. 
If you find that repository useful, please make a reference to it!

```bibtex
# Original Author
@InProceedings{pmlr-v80-ruff18a,
  title     = {Deep One-Class Classification},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {4393--4402},
  year      = {2018},
  volume    = {80},
}

# Mine
@misc{deep-anomaly-classification,
  title={Deep Anomaly Classification},
  author={Gabriel Dornelles Monteiro},
  year={2022},
  howpublished={GitHub Repository},
  url={https://github.com/GabrielDornelles/deep-anomaly-classification},
}

```