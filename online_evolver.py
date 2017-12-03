import numpy as np
import pickle as pkl
import neuralNet
import train_ESN
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

        print('Ready')

        while True:
            # the main loop
            # continues until process is killed

            # Real-time mode:
            # t = time.time()
            # if t > self.next_evo:
            #     self.next_evo = t + self.WAIT_TIME
            #     self.mutate()

            # Turbo Mode!
            try:
                log = np.loadtxt(open("log.csv", "rb"), delimiter=",")
                # when there are 10 or evaluations
                if log.size >=20:
                    self.mutate()
            except:
                pass

    #---------------------------------------------------------------------------
    #   EVOLVE!!
    #---------------------------------------------------------------------------

    def mutate(self):
        # load the log
        # this csv keeps track of evaluated neural networks
        log = np.loadtxt(open("./log.csv", "rb"), delimiter=",")
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
        print('\nCurrent Log:')
        print(log)

        # MUTATION OF BEST NN
        # first check if it still there...

        best_nn = self.PATH_TO_POOL + 'evesn%s.pkl'%int(log[0, 0])
        if not os.path.exists(best_nn):
            print('Best ESN not found :(')
        else:
            # find the best nn and compare to record:
            if log[0, 1] > self.current_record:
                best_id = int(log[0, 0])
                print('Best ID: ' + str(best_id))
                self.current_record = log[0, 1]
                copyfile(best_nn, self.PATH_TO_BEST + 'top_ESN.pkl')

            with open(best_nn, 'rb') as file:
                net = pkl.load(file, encoding='latin1')

            self.mutate_esn(net)
            # delete the worse nn
            try:
                # os.remove(log[-1, 0])
                os.remove(self.PATH_TO_POOL + 'evesn%s.pkl'%int(log[-1, 0]))
                print('Deleted evesn%s.pkl'%int(log[-1, 0]))
            except:
                print('Could not find network to delete')


        # and mutate the second best:
        second_nn = self.PATH_TO_POOL + 'evesn%s.pkl'%int(log[1, 0])
        if not os.path.exists(second_nn):
            print('Second best ESN not found')
        else:
            with open(second_nn, 'rb') as file:
                net2 = pkl.load(file, encoding='latin1')

            self.mutate_esn(net2)
            # delete the second worse nn
            try:
                os.remove(self.PATH_TO_POOL + 'evesn%s.pkl'%int(log[-2, 0]))
                print('Deleted evesn%s.pkl'%int(log[-2, 0]))
            except:
                print('Could not find network to delete')


        return


    def mutate_esn(self, net):
        # perturb the weights slightly with a small sigma
        sigma = 0.10
        rows, cols = net.w_reservoir.nonzero()
        for r, c in zip(rows, cols):
            net.w_reservoir += np.random.normal(0, sigma)

        print('Changed weights')

        # chance of adding or deleting a random number of nodes
        rand = np.random.random_sample()
        if rand < 0.01:
            dim = net.D_reservoir
            num = np.random.randint(1, 5)

            rows = np.random.choice(dim, num)
            cols = np.random.choice(dim, num)
            net.w_reservoir[rows, cols] = np.random.randn(num)

            print('Added %s weights'%num)
        elif rand < 0.02:
            num = np.random.randint(1, 5)
            rows, cols = net.w_reservoir.nonzero()
            nodes = np.random.choice(len(rows), num, replace=False)
            net.w_reservoir[rows[nodes], cols[nodes]] = 0
            print('Removed %s weights'%num)

        # retrain and save as a new network:
        train_ESN.train2(net.w_reservoir, self.PATH_TO_POOL + 'evesn%s.pkl'%self.generation)

        print('Created evesn%s.pkl'%self.generation)
        self.generation += 1

        return

# Initiate evolver and start it
god = Evolver()
god.start()
