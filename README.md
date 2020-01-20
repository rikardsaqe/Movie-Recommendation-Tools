# Movie-Recommendation-Tool

Built a Restricted Boltzmann Machine (RBM) architecture to predict movies a consumer would like with a ~76% accuracy.

Developed in the Spyder IDE using NumPy, Pandas and Scikit-learn for data processing, PyTorch for training/iterating on model.

The most difficult part of this project was understanding the intuition behind RBMs and implementing a class to create the 
architecture of the network and train it using K-step contrastive divergence.

Running built_rbm.py and importing the datasets within the first 35 lines of code should enable you to be able to plug and play
with it as well.

Steps to further improve on this model and skillset would be to train it on a larger dataset, expiriment with different objects 
of my RBM class to identify the optimal number of epochs, hidden nodes, and batch size, and implement techniques such as k 
means validation as I did for my previous project Customer Churn Predictor. 

This project was built as part of the Deep Learning A-Z Udemy course: https://www.udemy.com/course/deeplearning/
