#!/usr/bin/python3

import sys
import neuralNet
from neuralNet import EchoStateNet as ESN
import os
from os import path


"""
train a neural net with the given training data (as .csv file)
save the trained net at the given location

run the script in the terminal in the following way:
python training.py ./training_data/aalgorg.csv ./trained_nn/my_net.pkl

    - the first argument specifies the location of the training data to use. If it is a directory, all .csv files within this directory are used as training samples
    - the second argument determines the storing path for the trained network

"""


path_to_data = sys.argv[1]

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

net = ESN(D_in,D_out,teacher_forcing=True)
net.train(input_data, target_data, sys.argv[2])

print("Neural net trained and saved to " + sys.argv[2])
