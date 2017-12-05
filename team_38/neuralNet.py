#!/usr/bin/python3

import math
import pandas as pd
import numpy as np
import pickle as pkl
import time
from scipy.sparse import linalg

"""
load training data from a .csv file at the given location 'path_to_data'
if the file contains incomplete rows, these are ignored
Args:
    path_to_data: absolute or relative path to the .csv file that contains the training_data
Returns:
    input data and target output data as Tensors
"""
def load_training_data(path_to_data):

    # read csv file
    training_data = pd.read_csv(path_to_data,index_col=False, header=None)

    # split training dataframe into input and target output data
    # use the first 3 columns as target data, and the rest as input data
    input_data = training_data.iloc[:,3::]
    target_data = training_data.iloc[:,0:3]

    # combine acceleration and braking into one output [-1, 1]
    control = target_data.iloc[:, 0] - target_data.iloc[:, 1]
    target_data = pd.concat((control, target_data.iloc[:, 2]), axis=1)


    # check for missing values (nan entries) and delete detected rows
    nan_rows_input = input_data.isnull().any(1)
    nan_rows_target = target_data.isnull().any(1)
    nan_rows = np.where(np.logical_or(nan_rows_input,nan_rows_target))[0]
    input_data = input_data.drop(nan_rows)
    target_data = target_data.drop(nan_rows)

    return [input_data.values,target_data.values]



"""
Multi-layer perceptron with one hidden layers
Only the forward propagation is implemented, because the weights are learned using evolutionary computing
"""

class MLP():

    """
    Constructor
    """
    def __init__(self, D_in, D_h, D_out, w_1, w_2):
        self.D_in = D_in
        self.D_h = D_h
        self.D_out = D_out
        self.w_1 = w_1
        self.w_2 = w_2

    """
    predict output of given input x
    w_1 has shape (D_in+1,D_h) (+1 because bias vector is included in weight matrices)
    w_2 has shape (D_h+1,D_out)
    use tanh() as activation function for all layers
    """
    def predict(self,x):

        x = np.concatenate((np.ones(1),x))
        # Forward propagation
        y1 = x.dot(self.w_1)
        a1 = np.tanh(y1)
        a1 = np.concatenate((np.ones(1),a1))
        y2 = a1.dot(self.w_2)
        output = np.tanh(y2)

        return output
    """
    saves MLP object to .pkl file
    """
    def save(self, storage_path):

        with open(storage_path, 'wb') as file:
            pkl.dump(self,file)

"""
load MLP object from .pkl file
Args:
    path_to_net: location where the net has been saved to
Returns:
    MLP object
"""
def restore_MLP(path_to_net):

    with open(path_to_net, 'rb') as file:
        net = pkl.load(file, encoding='latin1')
        return net


class EchoStateNet():

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

    def __init__(self, D_in, D_out, D_reservoir=50,
                 spectral_radius=0.9, sparsity=0.8, teacher_forcing=True, reservoir_weights=None):

        # check for proper dimensionality of all arguments and write them down.
        self.D_in = D_in
        self.D_reservoir = D_reservoir
        self.D_out = D_out
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_state_ = np.random.RandomState(1)
        self.teacher_forcing = teacher_forcing

        self._init_weights()

        if reservoir_weights is None or reservoir_weights.shape != (self.D_reservoir,self.D_reservoir):
            self._init_reservoir()
        else:
            print('load weights')
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
        # compute the max eigenvalue
        rho = np.real(linalg.eigs(w, k=1,return_eigenvectors=False)[0])
        # if all eigenvalues are zero add connections to weight matrix until an eigenvalue != 0 is obtained
        while rho==0:
            index = self.random_state_.randint(0,self.D_reservoir,size=2)
            w[index[0],index[1]] = self.random_state_.rand(1) -0.5
            rho = np.real(linalg.eigs(w, k=1,return_eigenvectors=False)[0])
        # normalize and scale w to obtain a matrix with the desired spectral radius
        w = w*(self.spectral_radius/rho)
        self.w_reservoir = w
        #print(w[np.where(w!=0)])

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
            wash_out = 200 # HOW LARGE SHOULD THIS BE????
            extended_states = extended_states[:,wash_out:]
            target_outputs = target_outputs[:,wash_out:]

            # collect the extended_states for all training datasets i in the variable 'extended_states'
            if i==0:
                collected_extended_states = extended_states
                collected_target_outputs = target_outputs
            else:
                collected_extended_states = np.concatenate((collected_extended_states, extended_states), axis=1)
                collected_target_outputs = np.concatenate((collected_target_outputs, target_outputs), axis=1)




        # learn the readout weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        # determine W_out by solving the system w_out*extended_states = target_output
        # use the Moore-Penrose pseudoinverse of extended_states
        # w_out = target_output * pseudoinverse(extended_states)
        pseudoinverse = np.linalg.pinv(collected_extended_states)
        self.w_out = np.dot(np.arctanh(collected_target_outputs), pseudoinverse)


        # apply learned weights to the collected states
        pred_train = np.tanh(np.dot(self.w_out, collected_extended_states))

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

        if len(inputs.shape)>1:
            batch_size = inputs.shape[1]
            for i in range(batch_size):
                # assure that inputs are in the right shape
                if inputs[i].shape[0] != self.D_in:
                    inputs[i,:] = inputs[i,:].T
        else:
            batch_size = 1
            # assure that inputs are in the right shape
            if inputs.shape[0] != self.D_in:
                inputs = inputs.T

        if continuation:
            # make prediction based on the ESN history
            laststates = self.laststates
            lastinputs = self.lastinputs
            lastoutputs = self.lastoutputs
        else:
            # start from scratch without any memories
            laststates = np.zeros(self.D_reservoir)
            lastinputs = np.zeros(self.D_in)
            lastoutputs = np.zeros(self.D_out)

        if batch_size > 1:
            inputs = np.concatenate((lastinputs[:,None], inputs),axis=1)
        else:
            inputs = np.concatenate((lastinputs[:,None], inputs[:,None]),axis=1)

        states = np.concatenate((laststates[:,None], np.zeros((self.D_reservoir, batch_size))),axis=1)
        outputs = np.concatenate((lastoutputs[:,None], np.zeros((self.D_out, batch_size))),axis=1)


        # propagate inputs through the net
        for n in range(batch_size):
            states[:,n+1] = self._next_states(states[:,n], inputs[:,n+1], outputs[:,n])
            outputs[:,n+1] = np.tanh(np.dot(self.w_out, np.concatenate((inputs[:,n+1], states[:,n+1]))))

        # save last values to use them as starting point for the next prediction
        if continuation:
            self.laststates = states[:,-1]
            self.lastinputs = inputs[:,-1]
            self.lastoutputs = outputs[:,-1]

            if storage_path is not None:
                # save the trained network
                with open(storage_path, 'wb') as file:
                    pkl.dump(self,file)

        command = outputs[:,-batch_size:]
        return [command[0][0], command[1][0]]



    def load_w_out(self,path_to_w_out):

        with open(path_to_w_out, 'rb') as file:
            w_out = np.load(file)
            self.w_out = w_out

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