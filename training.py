#!/usr/bin/python3

import sys
import neuralNet
from neuralNet import EchoStateNet as ESN
from neuralNet import MultiLayerPerceptron as MLP
import os
from os import path
from pca import PCA as PCA


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


esn = True if sys.argv[1] == "esn" else False


path_to_data = sys.argv[2]

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

if esn:
    net = ESN(D_in,D_out,teacher_forcing=True)
else:
    net = MLP(D_in,D_in,D_out)

# PCA.train(path_to_data)

if len(sys.argv)>=4:

    if esn==False and len(sys.argv)==5:
        net.train(input_data, target_data, sys.argv[3], sys.argv[4])
        print("Initial network state loaded from " + sys.argv[3])
    else:
        net.train(input_data, target_data, sys.argv[3])
    print("Neural net trained and saved to " + sys.argv[3])

    #if esn:
    #    net.predict(input_data[0])
