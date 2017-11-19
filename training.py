#!/usr/bin/python3

import sys
import neuralNet

PATH_TO_TRAINING_DATA = "../train_data/data.csv"
STORAGE_PATH = "./trained_nn/net.pkl"

"""
train a neural net with the given training data (as .csv file)
save the trained net at the given location

run the script in the terminal in the following way
python training.py ../train_data/aalgorg.csv ./trained_nn/my_net.pkl

if no arguments are given, the default paths will be used
"""

#
if len(sys.argv)==3:

    path_to_training_data = sys.argv[1]
    input_data, target_data = neuralNet.load_training_data(path_to_training_data)
    print("Training data loaded from " + path_to_training_data)

    storage_path = sys.argv[2]
    neuralNet.train(input_data, target_data, storage_path)
    print("Neural net trained and saved to " + storage_path)

else:

    input_data, target_data = neuralNet.load_training_data(PATH_TO_TRAINING_DATA)
    print("Training data loaded from " + PATH_TO_TRAINING_DATA)
    neuralNet.train(input_data, target_data, STORAGE_PATH)
    print("Neural net trained and saved to " + STORAGE_PATH)
