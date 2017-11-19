import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import numpy as np

LEARNING_RATE = 1e-5
TRAINING_ITERATIONS = 1000


#class Net(torch.nn.Module):
#    def __init__(self, n_input, n_hidden, n_output):
#        super(Net, self).__init__()
#        self.hidden = torch.nn.Linear(n_input, n_hidden)   # hidden layer
#        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
#
#    def forward(self, x):
#        x = F.tanh(self.hidden(x))      # activation function for hidden layer
#        x = self.predict(x)             # linear output
#        return x

"""
load training data from the .csv file at the given location 'path_to_data'
if the file contains incomplete rows, these are ignored
RETURN input data and target output data as Tensors
"""
def load_training_data(path_to_data):

    # read csv file
    training_data = pd.read_csv(path_to_data,index_col=False)

    # split training dataframe into input and target output data
    # use the first 3 columns as target data, and the rest as input data
    input_data = training_data.iloc[:,3::]
    target_data = training_data.iloc[:,0:3]

    # check for missing values (nan entries) and delete detected rows
    nan_rows_input = input_data.isnull().any(1)
    nan_rows_target = target_data.isnull().any(1)
    nan_rows = np.where(np.logical_or(nan_rows_input,nan_rows_target))[0]
    input_data = input_data.drop(nan_rows)
    target_data = target_data.drop(nan_rows)

    # convert input and target output to Tensors
    input_data = torch.FloatTensor(input_data.values)
    target_data = torch.FloatTensor(target_data.values)

    return [input_data, target_data]

"""
train a 3 layer neural net with the given training data 'input_data' and 'target_data'
save the final model at the given location 'storage_path'
"""
def train(input_data, target_data, storage_path):

    # input dimension
    D_in = input_data.size(1)
    # hidden layer dimension
    D_h = D_in
    # output dimension
    D_out = target_data.size(1)

    # if using the nn.Module network
    #net = Net(n_input=D_in, n_hidden=D_h, n_output=D_out) # define the network

    # otherwise
    net = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_h),
    torch.nn.Tanh(),
    torch.nn.Linear(D_h, D_out),)

    # wrap inputs and outputs in variables
    x, y = Variable(input_data), Variable(target_data)

    # use Mean Squared Error (MSE) as loss function.
    loss_func = torch.nn.MSELoss(size_average=False)

    # set optimization algorithm (here: gradient descent)
    opt = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    for t in range(TRAINING_ITERATIONS):

        # Forward pass: compute predicted y by passing x to the model
        y_pred = net(x)

        # Compute loss
        loss = loss_func(y_pred, y)
        #print(loss)

        # Zero the gradients before running the backward pass
        opt.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model
        loss.backward()

        # update weights using gradient descent
        opt.step()


    # save the trained net
    torch.save(net, storage_path)  # save entire net
    #torch.save(net.state_dict(), 'net_params.pkl') # save only the parameters

"""
restore a previously trained neural net that has been saved at 'path_to_trained_net'
and apply it to the given input data
RETURN the resulting predictions
"""
def restore_net_and_predict(input,path_to_trained_net):

    # restore the entire neural net
    net = torch.load(path_to_trained_net)

    # print the learned parameters
    #for param in net.parameters():
    #    print(param.data, param.size())

    # predict output for the given input
    prediction = net(input)

    return prediction
