#!/usr/bin/python3

import sys
import neuralNet
from neuralNet import EchoStateNet as ESN
import os
from os import path
import datetime


"""
train an ESN with the given parameters:
D_reservoir: reservoir size
spectral_radius: should be smaller than one. Large values correspond to slower reservoir dynamics, small values to fast ones
sparsity: value in range [0,1)

the reservoir is created using the initialization scheme from the 'EchoStateNet' class
"""
def train1(D_reservoir,spectral_radius, sparsity):

    path_to_data = "./training_data/subset" # train on all data contained in the folder 'subset'. Change to .csv file if only one training sample should be used.
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
    print("train network with D_reservoir = %.0f, spectral_radius = %.4f, sparsity = %.4f" %(D_reservoir,spectral_radius,sparsity))

    net.train(input_data, target_data, storage_path)
    print("Neural net trained and saved")

"""
train an ESN with the given reservoir weight matrix 'w_reservoir'
"""
def train2(w_reservoir):

    path_to_data = "../training_data/subset" # train on all data contained in the folder 'subset'. Change to .csv file if only one training sample should be used.
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
    net.train(input_data, target_data, storage_path)

    print("Neural net trained and saved")
