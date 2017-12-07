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

class Evolver2():

    def __init__(self):
        self.PATH_TO_POOL = './evolver/pool/'
        self.PATH_TO_BEST = './evolver/best/'
        self.PATH_TO_DRIVERS_POOL = './evolver/driver_esn/'
        self.PATH_TO_QUEUE = './evolver/queue/'
        self.LOG = './log.csv'
        self.MIN_POOL_SIZE = 30
        self.WAIT_TIME = 10   # time to wait between mutations

        self.current_pool = []
        self.current_record = 0.0

    def start(self):
        t = time.time()
        self.next_evo = t + self.WAIT_TIME

        # # check if folders exist:
        # if not os.path.isdir(self.PATH_TO_POOL):
        #     print('NO POOL FOLDER FOUND \nTerminating...')
        #     return
        #
        # if not os.path.isdir(self.PATH_TO_BEST):
        #     print('Creating folder to keep best network...')
        #     os.system('mkdir %s'%self.PATH_TO_BEST)
        #
        # # load the current pool of neural networks:
        pool = os.listdir(self.PATH_TO_POOL)
        pool += os.listdir(self.PATH_TO_DRIVERS_POOL)

        if len(pool) == 0:
            if len(queue) > 0:
                self.gener
            print('NO INITIAL NETWORKS FOUND \nTerminating...')
            # self.generation =
            return
        else:
            for nn in pool:
                self.current_pool.append(int(nn[5:-4]))

            self.generation = max(self.current_pool) + 1

        print('Ready')

        while True:
            # main loop
            # maybe think of escape condition?

            drivers_pool = os.listdir(self.PATH_TO_DRIVERS_POOL)
            queue = os.listdir(self.PATH_TO_QUEUE)
            pool = os.listdir(self.PATH_TO_POOL)

            if len(queue) > 0:
                for esn in queue:
                    score = float(esn[:-4])
                    if len(pool) > self.MIN_POOL_SIZE:
                        # tournement select
                        log = np.loadtxt(open("./log.csv", "rb"), delimiter=",")
                        # log = log[log[:, 1].argsort()[::-1]]
                        num = log.shape[0]
                        tournement = log[np.random.choice(num, min(num, 5), replace=False), :]
                        # tournement = tournement[tournement[:, 1].argsort()[::-1]]

                        print('\nTournement:')
                        print(tournement)

                        # find loser and remove from /pool/ :
                        # loser_index = tournement.shape[0] - 1
                        loser_index = np.argmin(tournement[:, 1])
                        loser_id = int(tournement[loser_index, 0])

                        if score < tournement[loser_index, 1]:
                            print('No change...')
                            os.system('rm %s'%(self.PATH_TO_QUEUE + esn))
                            continue

                        print('\033[7mLoser: %s\033[0m'%loser_id)
                        os.system('rm %sevesn%s.pkl'%(self.PATH_TO_POOL, loser_id))
                    else:
                        print('Pool size increased')
                        loser_id = -1

                    # move from queue into the pool
                    os.system('mv %s %sevesn%s.pkl'%(self.PATH_TO_QUEUE + esn, self.PATH_TO_POOL, self.generation))

                    # rewrite log file and replace line with loser with new ESN
                    log_file = open('./log.csv', 'r')
                    log_new = open('./log_new.csv', 'w')
                    for line in log_file:
                        if int(line.split(',')[0]) == loser_id:
                            log_new.write('%s, %.2f\n'%(self.generation, score))
                        else:
                            log_new.write(line)

                    if loser_id == -1:
                        # append to the end of the log
                        log_new.write('%s, %.2f\n'%(self.generation, score))

                    log_file.close()
                    log_new.close()

                    os.system('mv ./log_new.csv ./log.csv')

                    self.generation += 1

            if (len(drivers_pool) < 3) and (len(pool) > 0):
                # no ESNs for the cars to use, mutate more
                log = np.loadtxt(open("./log.csv", "rb"), delimiter=",")
                choice = log[np.random.choice(log.shape[0], 1, replace=False), :][0]
                path = self.PATH_TO_POOL + 'evesn%s.pkl'%int(choice[0])
                self.mutate_esn(path, choice[1])




    #---------------------------------------------------------------------------
    #   EVOLVE!!
    #---------------------------------------------------------------------------


    def mutate_esn(self, path_to_net, fitness):
        # perturb the weights slightly with a small sigma
        # (make inversely proportional to the fitness!)
        with open(path_to_net, 'rb') as file:
            net = pkl.load(file, encoding='latin1')

        sigma = 0.2
        rows, cols = net.w_reservoir.nonzero()
        for r, c in zip(rows, cols):
            if np.random.random_sample() < 0.1:
                net.w_reservoir[r, c] += np.random.normal(0, sigma)

        # adjust in weights with 50% prob
        if np.random.random_sample() < 0.5:
            net.w_in += np.random.randn(net.w_in.shape[0], net.w_in.shape[1]) * 0.1

        # # adjust the out weights with 50% prob
        # if np.random.random_sample() < 0.5:
        #     net.w_out += np.random.randn(net_out.shape) * 0.1

        # adjust the back weights with 50% prob
        if np.random.random_sample() < 0.5:
            net.w_back += np.random.randn(net.w_back.shape[0], net.w_back.shape[1]) * 0.1

        print('Changed weights')

        # chance of adding or deleting a random number of nodes

        if np.random.random_sample() < 0.3:
            dim = net.D_reservoir
            num = np.random.randint(1, 10)

            rows = np.random.choice(dim, num)
            cols = np.random.choice(dim, num)
            net.w_reservoir[rows, cols] = np.random.randn(num)

            print('Added %s weights'%num)

        if np.random.random_sample() < 0.3:
            num = np.random.randint(1, 10)
            rows, cols = net.w_reservoir.nonzero()
            nodes = np.random.choice(len(rows), num, replace=False)
            net.w_reservoir[rows[nodes], cols[nodes]] = 0
            print('Removed %s weights'%num)

        # retrain and save as a new network:
        # train_ESN.train2(net.w_reservoir, self.PATH_TO_DRIVERS_POOL+ 'evesn%s.pkl'%self.generation)

        # save the network:
        # with open(self.PATH_TO_DRIVERS_POOL+ 'evesn%s.pkl'%self.generation, 'wb') as file:
        #     pkl.dump(self,file)

        net.save(self.PATH_TO_DRIVERS_POOL+ 'evesn%s.pkl'%self.generation)

        print('Created evesn%s.pkl'%self.generation)
        self.generation += 1

        return

# Initiate evolver and start it
god = Evolver2()
god.start()
