# Building a Restricted Boltzmann Machine 

# Importing libraries
import numpy as np  # used for working with arrays
import pandas as pd # import dataset, and create training and test set
import torch # import PyTorch
import torch.nn as nn # implement neural networks 
import torch.nn.parallel # # used for parallel computations
import torch.optim as optim # used for optimizer 
import torch.utils.data # tools
from torch.autograd import Variable # used for stochastic graident descent 

# Importing Data 

# Importing movie dataset (movie ids, names, genre) 
# Done by calling the path, what is used to seperate different pieces of data, the 
# header in the data, making sure the data is imported correctly, and an encoding due to special characters
# in movie titles 
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Importing User Dataset (user ids, gender, age, job id)
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Importing Movie Ratings (user ids, movie ids, movie ratings)
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Creating training set with user ids, movie ids, and ratings, containing 80% of total data 
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')

# Converting above dataframe into an array of integers so it's easier to work with
training_set = np.array(training_set, dtype = 'int') 

# Creating test set (20% of total data). Test set will have same users, but different movies
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')

# Converting above dataframe just like above
test_set = np.array(test_set, dtype = 'int') 

# Determining number of users by finding the maximum integer representing a user across all training and 
# test sets (because the data sets are all randomized we don't know where the biggest number is)
nb_users = int(max(max(training_set[:,0]), max(test_set[:, 0])))

# Determining number of movies just like above.
nb_movies = int(max(max(training_set[:,1]), max(test_set[:, 1])))

# Function converting dataset into a list of lists where each element holds all the movie ratings of an 
# individual user. In other words, an array of length u where u = total users, and where each element of 
# that array stores another array of length m where m stores the ratings the user gave each possible movie
# (again rating movies they didn't watch a 0).
def convert(data):
    new_data = [] # Initialize data
    for id_users in range(1, nb_users + 1): # for loop including from the first user to the last
        id_movies = data[:,1] [data[:,0] == id_users] # takes the all the watched movies of current id_user
        id_ratings = data[:,2] [data[:,0] == id_users] # takes the all the movie ratings of current id_user
        ratings = np.zeros(nb_movies) # creates array of zeros equal in size to total amount of movies
        ratings[id_movies - 1] = id_ratings # puts completed movie ratings into proper place in above array
        new_data.append(list (ratings)) # adds list created above to our main list as the first element
    return new_data # returns complete 2D array

# Calling the above function to convert our data sets into 2D arrays
training_set = convert(training_set)     
test_set = convert(test_set)     

# Making the data into Torch tensors (multi-dimensional matrix containing elements of a single data type)
# instead of what it was previously (a NumPy array)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network
class RBM():
    #Function that is required for all classes, allows us to initialize the object later. Takes in 
    # number of visible and hidden nodes.
    def __init__(self, nv, nh):
        # assigns weights of the probabilities of the visible nodes given the hidden nodes in a torch tensor
        # in a standard normal distribution 
        self.W = torch.randn(nh, nv) 
        # bias for the probability of the visible nodes given the hidden nodes. Has a 1 as its batch 
        # because it needs to be a 2D tensor
        self.a = torch.randn(1, nh) 
        self.b = torch.randn(1, nv) # bias for the probability of the hidden nodes given the visible nodes
   # Function that samples hidden nodes (guesses if they'll be activated/equal 1 or not) according to 
   # probabilities of hidden nodes given probabilites of visible nodes through the sigmoid activation function. 
   # This function is needed for performing gibbs sampling, which is what we will be using to approximate the 
   # log likelihood gradient which is what we use to optimize our RBM.
    def sample_h(self, x): # x is visible neurons 
        # product of 2 torch tensors weights and visible neurons. .t() finds the transpose, used to rearrange 
        # tensor into mathematically correct order.
        wx = torch.mm(x, self.W.t()) 
        activation = wx + self.a.expand_as(wx) # adds bias to all items of tensor product WX
        p_h_given_v = torch.sigmoid(activation) # applies activation function (sigmoid) to WX + b
        # Returns probability of hidden nodes activating given the visible nodes, and applies a bernouli
        # sampling to the values. Meaning: it will compare the probability of each hidden node to a random
        # number between 0 and 1, and if p_h_given_v is greater than that, the node is said to be activated.
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    # Does exact same thing as above function except samples visible nodes given hidden nodes.
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    # The common way to train the Boltzmann machine is to determine the parameters that maximize the likelihood 
    # of the observed data. To determine the parameters, we perform gradient descent on the log of the 
    # likelihood function. Goal is to maximize log likelihood/minimize energy. K step contrastive divergence 
    # (part of which is Gibs sampling) allows us to estimate log likelihood gradient 
    # (because doing the gradient directly is too computationally hard). This is the act of using the visible 
    # nodes weights to estimate the hidden nodes weights and using the hidden nodes weights to estimate the 
    # visible nodes weights and so on k times while also updating the biases.
    # Our function does this by taking in starting and ending weights for both visible and hidden nodes and 
    # going through the below operations.
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(ph0, v0 - torch.mm(phk, vk)) 
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
# Number of visible nodes is the amount of movies (initially)        
nv = len(training_set[0])

# Arbitrary number assigned to number of hidden nodes that can be changed to tune model and increase accuracy.
nh = 100

# Arbitrary number assigned to batch size that can be changed to tune model and increase accuracy.
batch_size = 100

# Creating object of RBM class
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10 # tuneable number
for epoch in range(1, nb_epoch + 1): # loops through all epochs
    train_loss = 0 # keeping track of difference between predicted ratings and actual ratings
    s = 0.   #counter used to normalize train loss
    # one loop takes you through amount of users in a batch
    for id_user in range(0, nb_users - batch_size, batch_size): 
        # values that are output of gibs sampling, returns predicted activations of visible nodes. Takes 
        # range of values from within the batch
        vk = training_set[id_user:id_user + batch_size]  
        v0 = training_set[id_user:id_user + batch_size]  # target value/initial value of visible nodes
        # probability hidden nodes equal 1 given visible input nodes. ,_ is used to only return first argument 
        # of sample_h, not bernoulli sample as well.
        ph0,_ = rbm.sample_h(v0)
        # Performing gibs sampling/k step contrastive divergence
        for k in range(10):
            _,hk = rbm.sample_h(vk) # gets first sampling of hidden nodes (only takes bernoulli sampling)
            _,vk = rbm.sample_v(hk) # gets first sampling of visible nodes (only takes bernoulli sampling)
            vk[v0<0] = v0[v0<0] # ensures all unwatched movies stay with value -1.
        phk,_ = rbm.sample_h(vk) # sampling of hidden nodes given last set of visible nodes
        rbm.train(v0, vk, ph0, phk) # training 
        # evalutes training loss by calculating the difference between real and predicted ratings
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) 
        s += 1. # increases counter
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s)) # outputs epoch number and training loss


# Testing the RBM. Similar code rationale to training.
test_loss = 0 
s = 0.
for id_user in range(nb_users): 
    v = training_set[id_user:id_user + 1] 
    vt = test_set[id_user:id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])) 
        s += 1.
print('test loss: '+str(test_loss/s))

