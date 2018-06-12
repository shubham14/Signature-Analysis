# Forged Signature Analysis
This project is aimed at forged signature analysis using a Tensorflow implementation 
of a deep architecture of a CNN-AE to find if the signature is forged or sone by the same person.

## Dependencies
Python - 3.6+
Tensorflow - 1.6.0+
numpy - 1.14.0+
keras - 2.1.4+

## Code Architecture
1. data_load.py - Processes the data(resize and grayscale the image)
2. model.py - Incorporates the CNN architecture 
3. squeeze_model.py - Incorporates the squeezenet model for training the model much quicker
4. VAE.py - Python implementation of the Variational Autoencoder for generating new samples of the signature
5. decision.py - Python implementation for one-class SVM to generate the decision boundary for the forged signature

## Authors
* **Shubham Dash** - [shubham14](https://github.com/shubham14)