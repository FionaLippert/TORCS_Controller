import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import numpy as np

# load training data
training_data = pd.read_csv("../train_data/aalborg.csv",index_col=False)

# use the first 3 columns as target data, and the rest as input data
# delete the last row because some values are missing
#input_data = training_data.iloc[:,3::].values[:-1,:]
#target_data = training_data.iloc[:,0:3].values[:-1,:]

# use only part of the dataset for training (here: time points 100 to 499)
input_data = training_data.iloc[100:500,3::].values
target_data = training_data.iloc[100:500,0:3].values

# use another part of the dataset for testing (here: time points 500 to 999)
test_input_data = torch.FloatTensor(training_data.iloc[500:1000,3::].values)
test_target_data = torch.FloatTensor(training_data.iloc[100:1000,0:3].values)

# convert input and target output to Tensors
input_data = torch.FloatTensor(input_data)
target_data = torch.FloatTensor(target_data)


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


def train_and_save_nn(input_data, target_data):

    # D_in is input dimension;
    # D_h is hidden dimension;
    # D_out is output dimension.
    D_in = input_data.size(1)
    D_h = D_in
    D_out = target_data.size(1)

    # if using the nn.Module network
    #net = Net(n_input=D_in, n_hidden=D_h, n_output=D_out) # define the network

    # otherwise
    net = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_h),
    torch.nn.Tanh(),
    torch.nn.Linear(D_h, D_out),
)

    # wrap inputs and outputs in variables
    x, y = Variable(input_data), Variable(target_data)

    # use Mean Squared Error (MSE) as loss function.
    loss_func = torch.nn.MSELoss(size_average=False)

    # set learning rate and optimization algorithm (here: gradient descent)
    learning_rate = 1e-5
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate)

    for t in range(1000):

        # Forward pass: compute predicted y by passing x to the model
        y_pred = net(x)

        # Compute loss
        loss = loss_func(y_pred, y)
        print(loss)

        # Zero the gradients before running the backward pass
        opt.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model
        loss.backward()

        # update weights using gradient descent
        opt.step()


    # 2 ways to save the net
    torch.save(net, 'net.pkl')  # save entire net
    torch.save(net.state_dict(), 'net_params.pkl') # save only the parameters


def restore_net_and_predict(input):

    # restore the entire neural net
    net = torch.load('net.pkl')

    # print the learned parameters
    #for param in net.parameters():
        #print(param.data, param.size())

    # predict output for the given input
    prediction = net(input)

    return prediction

train_and_save_nn(input_data,target_data)

sample_index = 0
prediction = restore_net_and_predict(Variable(test_input_data[sample_index]))
print prediction
print(test_target_data[sample_index])
