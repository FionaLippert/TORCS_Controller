# README

## my_driver.py
Contains the driving logic for the TORCS controller. It extends and partly overwrites the provided class `Driver()` in `pytocl/driver.py`
It implements

- the basic ESN driver, based on the `EchoStateNetwork()` class in `neuralNet.py`. Once the ESN has been loaded from the given .pkl file, it is saved as an property of the current `MyDriver()` instance to reduce file reading and writing costs.

- the simple recovery system

- the MLP controller extension, based on the `MLP()` class in `neuralNet.py`
	  (can be enabled and disabled by setting the variable 'use_mlp_opponents')

- communication and interaction between team-mates
	  (can be enabled and disables by setting the variable 'use_team')
	  All communication logs are read from and written to ./team_communication.

## neuralNet.py
Implements an ESN from scratch, solely reliant on numpy's and scipy's linear algebra functionalities.
For the MLP, only the forward pass was necessary to implement, because weights are learned through evolution.

## training.py
Basic script for ESN training. To generate a new network and train it on either one training sample (as .csv file) or a set of training data contained in a folder run:

`$ python training.py ./training_data/path/to/file/or/folder ./trained_nn/name_of_new_esn.pkl`


## start.sh
The main entry point for connecting the controller with the TORCS system.
It is linked to `run.py`.

- To run the offline evolution, please change in `run.py` the line 'from my_driver import MyDriver' to 'from my_driver_offline_evolution import MyDriver'.

- To run the online evolution, please change line 4 to 'from my_driver_evaluator import MyDriver'

## EA_offline.py
Contains the offline EA implementation based on the python library DEAP (F. Fortin, F. De Rainville and M. Gardner. DEAP: Evolutionary Algorithms Made Easy. Journal of Machine Learning Research, 13:2171--2175, 2012.)
For ESN reservoir evolution the method main() has to be used, and `main_MLP()` for MLP evolution.
EA evaluations are based on TORCS simulations executed by `simulate_and_evalutate.py`.
The best 10 solutions are saved together with a log and the final population into the folder ./EA_output.

## simulate_and_evaluate.py
Runs TORCS simulations with either only one car, against opponents, or as a team of two identical controllers. The config files for the respective TORCS races are loaded from the folder ./config_files.
It further provides methods to extract relevant data for fitness evaluation.

## train_ESN.py
Contains functions for ESN training that are used during the ESN reservoir offline evolution.

## my_driver_offline_evolution.py
A copy of `my_driver.py` reduced and modified for the offline evolution of ESN reservoirs. All relevant information for fitness determination is logged to a temporary file `simulation_log.txt`


## ./offline_evolution_EA/
Contains the results and logs of the offline evolution of reservoir weights.

## my_driver_evaluator.py
A copy of my_driver.py that picks an unevaluated network from ./evolver/driver_ESN/ to test for a given period. Fitness is calculated and the network renamed with this fitness and moved to ./evolver/queue/ for `online_evolver.py` to parse.

## online_evolver.py
Evolver script that takes a random network from ./evolver/pool/, mutates it and moves to ./evolver/driver_ESN/ for evaluation. Any network in ./evolver/queue/ is only introduced back into the gene pool if the evaluated fitness wins a tournament selection. The current gene pool is logged in `./log.csv` and evolution progress in `./progress.txt`

## run_evolutions.py
Script to continuously run TORCS (in text mode) as the evaluator cars race. Iterates over all race config files in ./evolver/race_config/ (each race has four cars, 100 laps). To be run concurrently to `online_evolver.py`

## run_car.py
Script to automatically start my_driver_evaluator.py after each race finishes. Unique port number has to be passed (e.g. `$ python run_car.py 3001`). Need to run four copies of this script at once for each driver in `run_evolutions.py`.

## ./evolver/
Contains the gene pool, race config files and folders where unevaluated and evaluated networks are queued.
