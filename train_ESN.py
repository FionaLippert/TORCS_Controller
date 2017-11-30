#!/usr/bin/python3

import sys
import neuralNet
from neuralNet import EchoStateNet as ESN
from neuralNet import MultiLayerPerceptron as MLP
import os
from os import path
import datetime


"""
train a neural net with the given training data (as .csv file)
save the trained net at the given location

run the script in the terminal in the following way:
python training.py esn ../training_data/aalgorg.csv ./trained_nn/my_net.pkl [./trained_nn/net.pkl]

    - the first argument specifies the network type to use. esn: echo state network, mlp: multi layer perceptron
    - the second argument specifies the location of the training data to use. If it is a directory, all .csv files within this directory are used
    - the third argument determines the storing path for the trained network
    - if the 4th argument is given, the corresponding trained network is used as starting point for parameter training (only for mlp!)

"""


def train1(D_reservoir,spectral_radius, sparsity):

    path_to_data = "../training_data/track_data/subset"
    #filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #storage_path = "./trained_nn/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".pkl"
    storage_path = "./trained_nn/esn.pkl"


    if path.isfile(path_to_data):
        input_data, target_data = neuralNet.load_training_data(path_to_data)
        D_in = len(input_data[0])
        D_out = len(target_data[0])
    else:
        # walk through directory and collect the data from all .csv files
        input_data = []
        target_data = []
        for file in os.listdir(path_to_data):
            if file.endswith(".csv"):
                input_data_i, target_data_i = neuralNet.load_training_data(path.join(path_to_data, file))
                input_data.append(input_data_i)
                target_data.append(target_data_i)

        D_in = len(input_data[0][0])
        D_out = len(target_data[0][0])

    print("Training data loaded from " + path_to_data)


    net = ESN(D_in,D_out,D_reservoir=D_reservoir,spectral_radius=spectral_radius,sparsity=sparsity,teacher_forcing=True)
    #net = MLP(D_in,D_in,D_out)
    print("train network with D_reservoir = %.0f, spectral_radius = %.4f, sparsity = %.4f" %(D_reservoir,spectral_radius,sparsity))
    net.train(input_data, target_data, storage_path)
    print("Neural net trained and saved")
    #print("learned weights: "+str(net.w_out))

    #return storage_path

def train2(w_reservoir):

    path_to_data = "../training_data/track_data/subset"
    #filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #storage_path = "./trained_nn/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".pkl"
    storage_path = "./trained_nn/esn.pkl"


    if path.isfile(path_to_data):
        input_data, target_data = neuralNet.load_training_data(path_to_data)
        D_in = len(input_data[0])
        D_out = len(target_data[0])
    else:
        # walk through directory and collect the data from all .csv files
        input_data = []
        target_data = []
        for file in os.listdir(path_to_data):
            if file.endswith(".csv"):
                input_data_i, target_data_i = neuralNet.load_training_data(path.join(path_to_data, file))
                input_data.append(input_data_i)
                target_data.append(target_data_i)

        D_in = len(input_data[0][0])
        D_out = len(target_data[0][0])

    print("Training data loaded from " + path_to_data)


    net = ESN(D_in,D_out,D_reservoir=w_reservoir.shape[0],reservoir_weights=w_reservoir,teacher_forcing=True)
    #net = MLP(D_in,D_in,D_out)
    #print("train network with D_reservoir = %.0f, spectral_radius = %.4f, sparsity = %.4f" %(D_reservoir,spectral_radius,sparsity))
    net.train(input_data, target_data, storage_path)
    #print(w_reservoir)
    print("Neural net trained and saved")
    #print("learned weights: "+str(net.w_out))
