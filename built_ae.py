# Implmenting a Stacked Autoencoder 

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

# Creating Neural Network Architecture

# The below uses inheritance. We will make class SAE a child class of existing class Module in PyTorch taken 
# from nn so that we can use all the variables and functions from Module.
class SAE(nn.Module):
    # Mandatory function used to initialize objects when making a class
    def __init__(self, ):
        # Super function is used to allow inheritance
        super(SAE, self).__init__()
        # Represents full connection between input nodes and the first set of encoded nodes. Second value of
        # 20 can be tuned to optimize network. Encoded nodes are used to try and find trends among our input
        # data, given that we are evaluating movies, a simple example of this could be genre.
        self.fc1 = nn.Linear(nb_movies, 20) 
        # Represents full connection between first and second set of encoded nodes.
        self.fc2 = nn.Linear(20, 10) 
        # Represents full connection between second set of encoded nodes and first layer of decoding nodes.
        self.fc3 = nn.Linear(10, 20) 
        # Represents full connection between second layer of decoding nodes and output layer. Autoencoders
        # have an equivalent amount of input and ouput nodes.
        self.fc4 = nn.Linear(20, nb_movies) 
        # Initializes activation function. This can be changed to attempt to tune the model to better results.
        self.activation = nn.Sigmoid()
    # Function that does the actual encoding and decoding of the different layers with the end resulting 
    # producing the vector of predicted ratings which we'll use to compare to actual ratings. It also applies 
    # the activation functions inside the full connections.
    def forward(self, x): # X is the input vector holding all the movie ratings given by a specific user.
        x = self.activation(self.fc1(x)) # reassigns x to be equal to the first encoded layer.
        x = self.activation(self.fc2(x)) # same idea as above
        x = self.activation(self.fc3(x)) # same idea as above except now x is being decoded.
        x = self.fc4(x) # activation not applied because this is the output layer.
        return x # returns predicted ratings

# Creates object of above class (which is our autoencoder)
sae = SAE()

# Define criterion object used for loss function (in this case mean squared error)
criterion = nn.MSELoss() 

# Define object used for optimizer (in this case rmsprop but that can be changed) that takens in all the 
# paramaters of autoencoders, learning rate (which can also be changed), and decay (used to decrease learning 
# rate across epochs to regulate the convergence of the values which can also be changed)
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200 # can be changed to tune model
for epoch in range(1, nb_epoch + 1): # loops through all epochs
    train_loss = 0
    s = 0. # number of users who rated at least one movie
    for id_user in range(nb_users): # loops through all actions done in one given epoch
        # takes all movie ratings of one given user. Input must be made into a 2D vector to be able to be
        # inputted into the functions in our class, so we need to create a "fake dimension" corresponding
        # to the batch of users of 1/index 0. You can choose to update weights after several users, which
        # would be batch learning, in which case this value would be important, however this is not our goal
        # as we want to update the weights after every individual user (this is online learning).
        input = Variable(training_set[id_user]).unsqueeze(0) 
        target = input.clone() # target = input initially, however x is going to be modified.
        # If condition used to optimize our memory during training by only evaluting users that have rated one 
        # movie or more. target.data stores all the information for the current user it's looping through.
        if torch.sum(target.data > 0) > 0:
            output = sae(input) # returns vector of predicted ratings
            target.require_grad = False # ensures gradient isn't computed with respect to target
             # saves memory during computation by not using unrated movies during computation
            output[target == 0] = 0
            loss = criterion(output, target) # computes loss
            # Calculates the average of the error of rated movies. 1e-10 is added to ensure the denominator is 
            # never 0, and therefore doesn't cause an error in calculating mean_corrector.
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # Determines whether or not we should increase or decrease weights.
            loss.backward()
            # Calculates updated training loss
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1. # updates counter value
            # Determines how much we should increase or decrease weights.
            optimizer.step()
    # returns epoch number and loss value. Note, the loss value shows how far off the prediction is from 
    # guessing the right rating from 1-5. A loss of 1 would mean the system is within the range of less than 1
    # star off of the correct answer.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s) )
    
    
# Testing the SAE. Similar logic to training the SAE just removing certain information/commands like    
# backpropogation and the epoch.
test_loss = 0
s = 0. # number of users who rated at least one movie
for id_user in range(nb_users): # loops through all actions done in one given epoch
    # takes all movie ratings of one given user. Input must be made into a 2D vector to be able to be
    # inputted into the functions in our class, so we need to create a "fake dimension" corresponding
    # to the batch of users of 1/index 0. You can choose to update weights after several users, which
    # would be batch learning, in which case this value would be important, however this is not our goal
    # as we want to update the weights after every individual user (this is online learning).
    # Note: we do not need to use test_set below even though we are testing right now
    input = Variable(training_set[id_user]).unsqueeze(0) 
    target = Variable(test_set[id_user]) # contains answers of our test set.
    # If condition used to optimize our memory during training by only evaluting users that have rated one 
    # movie or more. target.data stores all the information for the current user it's looping through.
    if torch.sum(target.data > 0) > 0:
           output = sae(input) # returns vector of predicted ratings
           target.require_grad = False # ensures gradient isn't computed with respect to target
           # saves memory during computation by not using unrated movies during computation
           output[(target == 0).unsqueeze(0)] = 0
           loss = criterion(output, target) # computes loss
           # Calculates the average of the error of rated movies. 1e-10 is added to ensure the denominator is 
           # never 0, and therefore doesn't cause an error in calculating mean_corrector.
           mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
           # Calculates updated test loss
           test_loss += np.sqrt(loss.item() * mean_corrector)
           s += 1. # updates counter value
# Returns loss value. Note, the loss value shows how far off the prediction is from 
# guessing the right rating from 1-5. A loss of 1 would mean the system is within the range of less than 1
# star off of the correct answer.
print('test loss: ' + str(test_loss/s))
        
        
    
    


        
        
        

        
        
    
    
    
    
    
    
    
