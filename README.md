# Movie-Recommendation-Tools

Built two models to predict movies a consumer would like. 

First, a Restricted Boltzmann Machine, with a ~76% accuracy at predicting binary results (the person either liked the movie or not).
Second, a Stacked Autoencoder, that predicted how much a user would like a movie given a rating of 1-5 with a less than 1 star difference (about ~0.95 start difference).

Both were developed in the Spyder IDE using NumPy, Pandas and Scikit-learn for data processing, PyTorch for training/iterating on model, and also involved the use of some object oriented programming principles like inheritance.

The most difficult part of these projects was understanding the intuition behind RBMs and implementing a class to create the 
architecture of the network and train it using K-step contrastive divergence.

Running built_rbm.py or built_ae and importing the datasets within the first 35 lines of code should enable you to be able to plug and play with it as well.

Steps to further improve on these models and skillset would be to train them on a larger dataset, expiriment with different objects of my RBM and AE classes to identify the optimal number of epochs, hidden nodes, and batch size, and implement techniques such as k means validation as I did for my previous project Customer Churn Predictor. 

These projects were built as part of the Deep Learning A-Z Udemy course: https://www.udemy.com/course/deeplearning/
