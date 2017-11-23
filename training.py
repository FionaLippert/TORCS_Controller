#!/usr/bin/python3

import sys
import neuralNet
from neuralNet import EchoStateNet as ESN
from neuralNet import MultiLayerPerceptron as MLP


"""
train a neural net with the given training data (as .csv file)
save the trained net at the given location

run the script in the terminal in the following way:
python training.py esn ../training_data/aalgorg.csv ./trained_nn/my_net.pkl [./trained_nn/net.pkl]

    - the first argument specifies the network type to use. esn: echo state network, mlp: multi layer perceptron
    - if the 4th argument is given, the corresponding trained network is used as starting point for parameter training

"""


esn = True if sys.argv[1] == "esn" else False

input_data, target_data = neuralNet.load_training_data(sys.argv[2])
D_in = len(input_data[0])
D_out = len(target_data[0])
print("Training data loaded from " + sys.argv[2])

if esn:
    net = ESN(D_in,D_out,teacher_forcing=False)
else:
    net = MLP(D_in,D_in,D_out)

if len(sys.argv)>=4:

    if esn==False and len(sys.argv)==5:
        net.train(input_data, target_data, sys.argv[3], sys.argv[4])
        print("Initial network state loaded from " + sys.argv[3])
    else:
        net.train(input_data, target_data, sys.argv[3])
    print("Neural net trained and saved to " + sys.argv[3])

    #if esn:
    #    net.predict(input_data[0])
