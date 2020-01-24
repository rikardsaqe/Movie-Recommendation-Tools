# Movie-Recommendation-Tools

Built two models to predict movies a consumer would like:
- First, a Restricted Boltzmann Machine, with a ~76% accuracy at predicting binary results (the person either liked the movie or not).
- Second, a Stacked Autoencoder that predicted how much a user would like a movie given a rating of 1-5 with a less than 1 star difference (about ~0.95 start difference).

# Getting Started
- Download built_rbm.py, built_ae, ml-100k, and ml-1m into the same folder
- Run built_rbm.py and/or built_ae

# Built With
- **The environment:** [Spyder IDE](https://www.spyder-ide.org/)
- **Data Processing:** [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Scikit-learn](https://scikit-learn.org/stable/index.html) 
- **Model Training/Iterating:** [Pytorch](https://pytorch.org/)
- **Programming Concepts:** Used some object oriented programming principles like inheritance.
- **Education:** Part of the [Deep Learning A-Z Udemy course](https://www.udemy.com/course/deeplearning/)

# Challenges
The most difficult part of these projects was understanding the intuition behind RBMs and implementing a class to create the 
architecture of the network and train it using K-step contrastive divergence.

# Next Steps For Improvement
- Train both algorithms on a larger dataset from [MovieLens](https://grouplens.org/datasets/movielens/) 
- Expiriment with different objects of my RBM and AE classes to identify the optimal number of epochs, hidden nodes, and batch size 
- Implement techniques such as k means validation as I did for my previous project Customer Churn Predictor 

# Author
- **Rikard Saqe** [Github](https://github.com/rikardsaqe/)

# License
- This project is licensed under the MIT License, see the [LICENSE.txt](https://github.com/rikardsaqe/Movie-Recommendation-Tools/blob/master/LICENSE) file for details
