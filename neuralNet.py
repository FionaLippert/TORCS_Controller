#!/usr/bin/python3

import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle as pkl
import time
from pca import PCA

"""
load training data from a .csv file at the given location 'path_to_data'
if the file contains incomplete rows, these are ignored
Args:
    path_to_data: absolute or relative path to the .csv file that contains the training_data
Returns:
    input data and target output data as Tensors
"""
def load_training_data(path_to_data):
    use_pca = True
    # read csv file
    training_data = pd.read_csv(path_to_data,index_col=False, header=None)

    # split training dataframe into input and target output data
    # use the first 3 columns as target data, and the rest as input data
    input_data = training_data.iloc[:,3::]
    target_data = training_data.iloc[:,0:3]

    # combine acceleration and braking into one output [-1, 1]
    control = target_data.iloc[:, 0] - target_data.iloc[:, 1]
    target_data = pd.concat((control, target_data.iloc[:, 2]), axis=1)

    if use_pca:
        car_data = pd.DataFrame(input_data.iloc[:, :3], columns=None, index=None)

        range_data = input_data.iloc[:, 3:]
        reduced_data = pd.DataFrame(PCA.convert(range_data), columns=None, index=None)
        input_data = pd.concat((car_data, reduced_data), axis=1)


    # check for missing values (nan entries) and delete detected rows
    nan_rows_input = input_data.isnull().any(1)
    nan_rows_target = target_data.isnull().any(1)
    nan_rows = np.where(np.logical_or(nan_rows_input,nan_rows_target))[0]
    input_data = input_data.drop(nan_rows)
    target_data = target_data.drop(nan_rows)

    return [input_data.values,target_data.values]

class EchoStateNet():

    """
    Evolve: reservoir size, spectral_radius, sparsity. Or: reservoir weights with given reservoir size
    Evolve with opponents: evolve output weigths
    """

    """
    Constructor

    creates an echo state network with
    D_in: input dimensionality
    D_out: output dimensionality
    D_reservoir: number of neurons in the reservoir
    spectral_radius: largest eigenvalue of the randomly initialized reservoir weight matrix
    sparsity: ratio of zero entries in the reservoir weight matrix
    teacher_forcing: use feedback from the previous output?
    """

    # Decent settings: resv=50, spars=0.8, radius=0.9

    def __init__(self, D_in, D_out, D_reservoir=20,
                 spectral_radius=0.5, sparsity=0.5, teacher_forcing=True, reservoir_weights=None):

        # check for proper dimensionality of all arguments and write them down.
        self.D_in = D_in
        self.D_reservoir = D_reservoir
        self.D_out = D_out
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_state_ = np.random.RandomState(1)
        self.teacher_forcing = teacher_forcing

        if reservoir_weights is None or reservoir_weights.shape != (self.D_reservoir,self.D_reservoir):
            self._init_weights()
            self._init_reservoir()
        else:
            self.w_reservoir = reservoir_weights

        self.laststates = np.zeros(self.D_reservoir)
        self.lastinputs = np.zeros(self.D_in)
        self.lastoutputs = np.zeros(self.D_out)

    """
    initialize input weights 'w_in' and feedback weights 'w_back', reservoir weights 'w_reservoir'
    """
    def _init_weights(self):

        # Initialize input weights randomly within range [-1,1]
        self.w_in = (self.random_state_.rand(self.D_reservoir, self.D_in) - 0.5) * 2.0

        # Initialize feedback weights randomly within range [-1,1]
        if self.teacher_forcing:
            self.w_back = (self.random_state_.rand(self.D_reservoir, self.D_out) - 0.5) * 2.0

    """
    initialize reservoir weights 'w_reservoir'
    """
    def _init_reservoir(self):
        # initialize reservoir weights
        # create random weights centered around zero
        w = self.random_state_.rand(self.D_reservoir, self.D_reservoir) - 0.5
        # force the weight matrix to be sparse: set weights with probability=sparsity to zero
        w[self.random_state_.rand(self.D_reservoir, self.D_reservoir) < self.sparsity] = 0
        # normalize and scale w to obtain a matrix with the desired spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(w)))
        w = w*(self.spectral_radius/rho)
        self.w_reservoir = w

    """
    perform one update step of the network
    use activation function tanh()
    Args:
        previous_state: states at time point (n-1), array of length D_reservoir
        current_input: input for time point n, array of length D_input
        previous_output: network output at time point (n-1), array of length D_out
    Returns:
        network states for time point n
    """
    def _next_states(self, previous_state, current_input, previous_output):

        if self.teacher_forcing:
            return np.tanh(np.dot(self.w_in,current_input) + np.dot(self.w_reservoir, previous_state) + np.dot(self.w_back, previous_output))
        else:
            return np.tanh(np.dot(self.w_in,current_input) + np.dot(self.w_reservoir, previous_state))


    """
    drive the network by the training data
    Args:
        inputs: input data, list of independent input data sets as 2D-arrays of shape (D_in, number of training samples)
        target_outputs: corresponding desired output values, list of independent output data sets as 2D-array of shape (D_out, number of training samples)
        storage_path: location to save the trained EchoStateNet object to
    Returns:
        network output for the training input data using the learned readout weights
    """
    def train(self, all_inputs, all_target_outputs, storage_path):

        collected_extended_states = np.empty(0)
        collected_target_outputs = np.empty(0)

        if type(all_inputs) is not list:
            all_inputs = [all_inputs]
            all_target_outputs = [all_target_outputs]

        # loop over all independent training datasets
        for i in range(len(all_inputs)):

            # assure that outputs are <1 and >-1
            inputs = np.asarray(all_inputs[i])
            target_outputs = np.asarray(all_target_outputs[i])
            outputs_too_large = np.where(target_outputs>=1)
            target_outputs[outputs_too_large] = 0.99999
            outputs_too_small = np.where(target_outputs<=-1)
            target_outputs[outputs_too_small] = -0.99999

            # assure that inputs and outputs are in the right shape
            if inputs.shape[0] != self.D_in:
                inputs = inputs.T
            if target_outputs.shape[0] != self.D_out:
                target_outputs = target_outputs.T
            batch_size = inputs.shape[1]

            # step the reservoir through the given input,output pairs
            # for each independent training dataset start with zero-states again
            states = np.zeros((self.D_reservoir, batch_size))
            for n in range(1, batch_size):
                states[:,n] = self._next_states(states[:,n-1], inputs[:,n], target_outputs[:,n-1])


            # concatenation of inputs u(n) and states x(n) --> get matrix with shape (D_in + D_reservoir, time points T)
            extended_states = np.concatenate((inputs, states), axis=0)

            # ignore the first t_0 time steps (wash out time) for the computation of output weights
            wash_out = 5 # HOW LARGE SHOULD THIS BE????
            extended_states = extended_states[:,wash_out:]
            target_outputs = target_outputs[:,wash_out:]

            # collect the extended_states for all training datasets i in the variable 'extended_states'
            if i==0:
                collected_extended_states = extended_states
                collected_target_outputs = target_outputs
            else:
                collected_extended_states = np.concatenate((collected_extended_states, extended_states), axis=1)
                collected_target_outputs = np.concatenate((collected_target_outputs, target_outputs), axis=1)
            print('shape of extended_states: '+str(collected_extended_states.shape))




        # learn the readout weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        # determine W_out by solving the system w_out*extended_states = target_output
        # use the Moore-Penrose pseudoinverse of extended_states
        # w_out = target_output * pseudoinverse(extended_states)
        pseudoinverse = np.linalg.pinv(collected_extended_states)
        self.w_out = np.dot(np.arctanh(collected_target_outputs), pseudoinverse)

        # remember the last state for later:
        #self.laststate = states[:,-1]
        #self.lastinput = inputs[:,-1]
        #self.lastoutput = target_outputs[:,-1]

        #self.laststates = np.zeros((self.D_reservoir,50))
        #self.lastinputs = np.zeros((self.D_in,50))
        #self.lastoutputs = np.zeros((self.D_out,50))

        # apply learned weights to the collected states and report the mean squared error
        pred_train = np.tanh(np.dot(self.w_out, collected_extended_states))
        print(np.sqrt(np.mean((pred_train - np.arctanh(collected_target_outputs))**2)))

        # save the trained network
        with open(storage_path, 'wb') as file:
            pkl.dump(self,file)

        return pred_train

    """
    Apply the trained network to unseen input data
    Args:
        inputs: input data, array of shape (D_in, number of samples (signal length))
        continuation: should the network be started from the last training or prediction state? otherwise: start with zeros
    Returns:
        predicted outputs, array of shape (D_out, number of samples (signal length))
    """
    def predict(self, inputs, continuation=True, storage_path=None):

        inputs = np.asarray(inputs)

        # assure that inputs are in the right shape
        # if inputs.shape[0] != self.D_in:
        #     inputs = inputs.T
        if len(inputs.shape)>1:
            batch_size = inputs.shape[1]
        else:
            batch_size = 1

        if continuation:
            laststates = self.laststates
            lastinputs = self.lastinputs
            lastoutputs = self.lastoutputs
        else:
            laststates = np.zeros(self.D_reservoir)
            lastinputs = np.zeros(self.D_in)
            lastoutputs = np.zeros(self.D_out)

        if batch_size > 1:
            #inputs = np.concatenate((lastinputs, inputs),axis=1)
            inputs = np.concatenate((lastinputs[:,None], inputs),axis=1)
        else:
            inputs = np.concatenate((lastinputs[:,None], inputs[:,None]),axis=1)

            #inputs = np.stack((lastinputs, inputs),axis=-1)
        states = np.concatenate((laststates[:,None], np.zeros((self.D_reservoir, batch_size))),axis=1)
        outputs = np.concatenate((lastoutputs[:,None], np.zeros((self.D_out, batch_size))),axis=1)
        #states = np.concatenate((laststates, np.zeros((self.D_reservoir, batch_size))),axis=1)
        #outputs = np.concatenate((lastoutputs, np.zeros((self.D_out, batch_size))),axis=1)


        for n in range(batch_size):
            states[:,n+1] = self._next_states(states[:,n], inputs[:,n+1], outputs[:,n])
            outputs[:,n+1] = np.tanh(np.dot(self.w_out, np.concatenate((inputs[:,n+1], states[:,n+1]))))

        # save last values to use them as starting point for the next prediction
        if continuation:
            self.laststates = states[:,-1]
            self.lastinputs = inputs[:,-1]
            self.lastoutputs = outputs[:,-1]

            #if storage_path is not None:
            #    # save the trained network
            #    with open(storage_path, 'wb') as file:
            #        pkl.dump(self,file)

        command = outputs[:,-batch_size:]
        return [command[0][0], command[1][0]]

"""
load EchoStateNet object from .pkl file and use it to predict outputs
Args:
    inputs: input data, array of shape (D_in, number of samples (signal length))
    path_to_trained_net: location where the trained net has been saved to
    continuation: should the network be started from the last training or prediction state? otherwise: start with zeros
Returns:
    predicted outputs, array of shape (D_out, number of samples (signal length))
"""
def restore_ESN_and_predict(inputs, path_to_trained_net, continuation=True):

    with open(path_to_trained_net, 'rb') as file:
        net = pkl.load(file, encoding='latin1')
        return net.predict(inputs, continuation, storage_path=path_to_trained_net)


"""
load EchoStateNet object from .pkl file
Args:
    path_to_trained_net: location where the trained net has been saved to
Returns:
    EchoStateNet object
"""
def restore_ESN(path_to_trained_net):

    with open(path_to_trained_net, 'rb') as file:
        net = pkl.load(file, encoding='latin1')
        return net


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


class MultiLayerPerceptron():

    """
    Constructor
    """
    def __init__(self, D_in, D_h, D_out, learning_rate=1e-6, n_iterations=10000):
        self.D_in = D_in
        self.D_h = D_h
        self.D_out = D_out
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.net = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_h),
        torch.nn.Tanh(),
        torch.nn.Linear(D_h, D_out),)


    """
    train a 3 layer neural net using gradient descent, save the resulting model
    Args:
        input_data: training input data
        target_data: desired outputs
        storage_path: location to save the final model to
        path_to_initial_net: if given, perform the training starting from the weights of this net, otherwise start with random weights
    """
    def train(self, input_data, target_data, storage_path, path_to_initial_net=None):

        if type(input_data) is list:
            input_data = np.concatenate(tuple(input_data), axis=0)
            target_data = np.concatenate(tuple(target_data), axis=0)


        # convert input and target output to Tensors
        input_data = torch.FloatTensor(input_data)
        target_data = torch.FloatTensor(target_data)

        # if an initial net is given, use this as starting point for parameter training
        if path_to_initial_net is not None:
            self.net = torch.load(path_to_initial_net)

        # assure that shapes are is correct
        if input_data.size(1) != self.D_in:
            input_data = input_data.T
        if target_data.size(1) != self.D_out:
            target_data = target_data.T

        #for param in self.net.parameters():
        #    print(param.data)

        # wrap inputs and outputs in variables
        x, y = Variable(input_data), Variable(target_data)

        # use Mean Squared Error (MSE) as loss function.
        loss_func = torch.nn.MSELoss(size_average=False)

        # set optimization algorithm (here: gradient descent)
        opt = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.5)

        for t in range(self.n_iterations):

            # Forward pass: compute predicted y by passing x to the model
            y_pred = self.net(x)

            # Compute loss
            loss = loss_func(y_pred, y)
            print(loss.values)

            # Zero the gradients before running the backward pass
            opt.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model
            loss.backward()

            # update weights using gradient descent
            opt.step()


        # save the trained net
        torch.save(self.net, storage_path)  # save entire net
        #torch.save(self.net.state_dict(), 'net_params.pkl') # save only the parameters

"""
restore a previously trained neural net and make a prediction
Args:
    input: input data for the prediction
    path_to_trained_net: path to the saved net (as .pkl file)
Returns:
    the resulting prediction
"""
def restore_MLP_and_predict(input_data,path_to_trained_net):

    # convert input and target output to Tensors
    input_data = Variable(torch.FloatTensor(input_data))

    # restore the entire neural net
    net = torch.load(path_to_trained_net)

    # print the learned parameters
    #for param in net.parameters():
    #    print(param.data, param.size())

    # predict output for the given input
    prediction = net(input_data)

    return prediction.data
