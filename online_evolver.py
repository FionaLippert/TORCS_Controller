import numpy as np
import pickle as pkl
import neuralNet
import time
import os
from shutil import copyfile

'''
Script that runs in parallel to TORCS which evolves evaluated ESN as they are
being tested.

Need a folder './pool/' where the population of ESNs are stored, where each
file has the name 'evesn####.pkl' where #### is the I.D. of that network
(I.D. can be any length integer).

Also need 'log.csv' where the cars will log their evaluations

And a folder './best/' where this program will store the highest rated ESN

Just start this from terminal with the command:
    $ python online_evolver.py

To terminate, kill process with ctrl+c
'''

class Evolver():

    def __init__(self):
        self.PATH_TO_POOL = './pool/'
        self.PATH_TO_BEST = './best/'
        self.WAIT_TIME = 10   # time to wait between mutations

        self.current_pool = []
        self.current_record = 0.0


    def start(self):
        t = time.time()
        self.next_evo = t + self.WAIT_TIME

        # check if folders exist:
        if not os.path.isdir(self.PATH_TO_POOL):
            print('NO POOL FOLDER FOUND \nTerminating...')
            return

        if not os.path.isdir(self.PATH_TO_BEST):
            print('Creating folder to keep best network...')
            os.system('mkdir %s'%self.PATH_TO_BEST)

        # load the current pool of neural networks:
        pool = os.listdir('./pool')
        if len(pool) == 0:
            print('NO INITIAL NETWORKS FOUND \nTerminating...')
            return

        for nn in pool:
            self.current_pool.append(int(nn[5:-4]))

        self.generation = max(self.current_pool) + 1

        # print('STARTING...')
        # while time.time() - t < 3.0:
        #     # pause for 5 seconds to sync with cars
        #     pass

        print('Ready')

        while True:
            # the main loop
            # continues until process is killed
            t = time.time()
            if t > self.next_evo:
                self.next_evo = t + self.WAIT_TIME
                # print('MUTATION')
                self.mutate()

    #---------------------------------------------------------------------------
    #   EVOLVE!!
    #---------------------------------------------------------------------------

    def mutate(self):
        # load the log
        # this csv keeps track of evaluated neural networks
        log = np.loadtxt(open("log.csv", "rb"), delimiter=",")
        # print(log)
        if log.size == 0:
            print('Log file empty...')
            return

        # sort the log file by best fitness:
        if log.size <= 8:
            # we need at least 4 evaluations to proceed
            # not enough evaluatoins to proceed
            print('Not enough evaluations...')
            return

        # clear the log:
        log_file = open('./log.csv', 'w')
        log_file.write('')
        log_file.close()
        # print('cleared log!')

        log = log[log[:, 1].argsort()[::-1]]
        print('Current Log:')
        print(log)

        # MUTATION OF BEST NN
        # first check if it still there...

        best_nn = self.PATH_TO_POOL + 'evesn%s.pkl'%int(log[0, 0])
        if not os.path.exists(best_nn):
            print('Best ESN not found :(')
            return

        # find the best nn and compare to record:
        if log[0, 1] > self.current_record:
            best_id = int(log[0, 0])
            print('Best ID: ' + str(best_id))
            self.current_record = log[0, 1]
            # path = self.PATH_TO_POOL + 'evesn%s.pkl'%best_id
            copyfile(best_nn, self.PATH_TO_BEST + 'top_ESN.pkl')

        with open(best_nn, 'rb') as file:
            net = pkl.load(file, encoding='latin1')

        if np.random.random_sample() < 0.05:
            # add a new neuron
            dim = net.D_reservoir
            row, col = np.random.randint(dim, size=2)

            if net.w_reservoir[row, col] == 0:
                net.w_reservoir[row, col] = np.random.randn()
            else:
                net.w_reservoir[row, col] = 0

            print('Added new neuron')
        else:
            # change a few existing neurons
            # for i in range(3):
            row, col = np.nonzero(net.w_reservoir)
            weights = np.random.choice(len(row), 3, replace=False)
            net.w_reservoir[row[weights], col[weights]] = np.random.randn(3)
            print('Changed weights')

        # save as a new network:
        with open(self.PATH_TO_POOL + 'evesn%s.pkl'%self.generation, 'wb') as file:
            pkl.dump(net,file)
        print('Created evesn%s.pkl'%self.generation)
        self.generation += 1

        # delete the worse nn
        try:
            # os.remove(log[-1, 0])
            os.remove(self.PATH_TO_POOL + 'evesn%s.pkl'%int(log[-1, 0]))
            print('Deleted evesn%s.pkl'%int(log[-1, 0]))
        except:
            print('Could not find network to delete')


# Initiate evolver and start it
god = Evolver()
god.start()
