#!/usr/bin/python3

import sys
import neuralNet

PATH_TO_TRAINING_DATA = "../train_data/data.csv"
STORAGE_PATH = "./trained_nn/net.pkl"

"""
train a neural net with the given training data (as .csv file)
save the trained net at the given location

run the script in the terminal in the following way:
python training.py ../train_data/aalgorg.csv ./trained_nn/my_net.pkl [./trained_nn/net.pkl]

    - if the third argument is given, the corresponding trained network is used as starting point for parameter training
    - if no arguments are given, the default paths will be used
"""

#
if len(sys.argv)>=3:

    input_data, target_data = neuralNet.load_training_data(sys.argv[1])
    print("Training data loaded from " + sys.argv[1])

    storage_path = sys.argv[2]
    if len(sys.argv)==4:
        neuralNet.train(input_data, target_data, sys.argv[2], sys.argv[3])
        print("Initial network state loaded from " + sys.argv[3])
    else:
        neuralNet.train(input_data, target_data, sys.argv[2])
    print("Neural net trained and saved to " + sys.argv[2])

else:

    input_data, target_data = neuralNet.load_training_data(PATH_TO_TRAINING_DATA)
    print("Training data loaded from " + PATH_TO_TRAINING_DATA)
    neuralNet.train(input_data, target_data, STORAGE_PATH)
    print("Neural net trained and saved to " + STORAGE_PATH)
