#!/usr/bin/python3
import random
from deap import base,creator,tools,algorithms,cma
import numpy
import simulate_and_evaluate
import train_ESN
import traceback
import datetime
import pickle
import math

from itertools import repeat
from collections import Sequence
import neuralNet

"""
Evolutionary Algorithm
"""

D_RESERVOIR = 50
D_IN = 22
D_OUT = 2

D_IN_MLP = 38
D_H_MLP = 20
D_OUT_MLP = 2

evolve_reservoir = False
evolve_mlp = True


"""
evaluation of ESN reservoir weights
"""
def evaluate(individual):

    # convert list of weights to a square matrix
    w_reservoir = numpy.asarray(individual).reshape((D_RESERVOIR,D_RESERVOIR))

    # training
    try:
        train_ESN.train2(w_reservoir)
    except Exception as err:
        # write error info to file
        with open('./EA_output/error_log.txt', 'a') as f:
            f.write('-------'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'-------------\n')
            f.write(str(err))
            traceback.print_exc(file=f)

    # simulate race on 3 different tracks
    dist = 0
    dist_from_center = 0
    stopped_all = 0
    offroad_all = 0

    for i in range(3):
        try:
            simulate_and_evaluate.simulate_track(i)
            d, d_from_center, stopped, offroad, angle = simulate_and_evaluate.get_fitness_after_time(60.0)

        except Exception as err:
            # write error info to file
            with open('./EA_output/error_log.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                f.write(str(err))
                traceback.print_exc(file=f)
            # penalty
            d = -10


        dist += d
        dist_from_center += d_from_center
        stopped_all += stopped
        offroad_all += offroad

        print("distance raced: %.2f"%d)
        print("accumulated distances from center: %.2f"%d_from_center)
        print("stopped: %.0f"%stopped)
        print("offroad: %.0f"%offroad)
        print("angle: %.2f"%angle)


    return (0.1*dist_from_center+100*stopped_all+100*offroad_all+0.01*angle-dist,)


"""
evaluation of ESN readout weights
"""
def evaluate_w_out(individual):

    # convert list of weights to a square matrix
    w_out = numpy.asarray(individual).reshape((D_OUT,D_RESERVOIR+D_IN))
    numpy.save('./w_out.npy',w_out)


    # simulate race on 3 different tracks
    dist = 0
    dist_from_center = 0
    stopped_all = 0
    offroad_all = 0
    angle_all = 0

    for i in range(3):
        try:
            simulate_and_evaluate.simulate_track(i)
            d, d_from_center, stopped, offroad, angle = simulate_and_evaluate.get_fitness_after_time(60.0)

        except Exception as err:
            # write error info to file
            with open('./EA_output/error_log.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                f.write(str(err))
                traceback.print_exc(file=f)
            # penalty
            d = -10

        dist += d
        dist_from_center += d_from_center
        stopped_all += stopped_all
        offroad_all += offroad
        angle_all += angle

        print("distance raced: %.2f"%d)
        print("accumulated distances from center: %.2f"%d_from_center)
        print("stopped: %.0f"%stopped)
        print("offroad: %.0f"%offroad)
        print("angle: %.2f"%angle)


    return (0.1*dist_from_center+100*stopped_all+100*offroad_all+0.01*angle-dist,)


"""
evaluation of MLP in a simulation with two cars racing against each other
"""
def evaluate_MLP(individual):

    w_1 = numpy.asarray(individual[:(D_IN_MLP+1)*D_H_MLP]).reshape((D_IN_MLP+1,D_H_MLP))
    w_2 = numpy.asarray(individual[(D_IN_MLP+1)*D_H_MLP:]).reshape((D_H_MLP+1,D_OUT_MLP))
    mlp = neuralNet.MLP(D_IN_MLP,D_H_MLP,D_OUT_MLP,w_1,w_2)
    mlp.save("./trained_nn/mlp_opponents.pkl")


    # simulate race on 3 different tracks
    fitness = 0
    for i in range(3):
        try:
            simulate_and_evaluate.simulate_track_two_cars(i)
            overtaking, opponents, dist_from_center, damage, MLP_accelerator_dev, MLP_steering_dev = simulate_and_evaluate.get_fitness_after_time(60.0, mlp=True)
            fitness += (0.01*dist_from_center + 0.01*opponents + damage + 0.01*MLP_steering_dev + 0.01*MLP_accelerator_dev - 100*overtaking)

            print("overtaking: %.0f"%overtaking)
            print("accumulated distances from center: %.2f"%dist_from_center)
            print("opponents: %.2f"%opponents)
            print("damage: %.0f"%damage)
            print("accelerator dev: %.2f"%MLP_accelerator_dev)
            print("steering dev: %.2f"%MLP_steering_dev)

        except Exception as err:
            # write error info to file
            with open('./EA_output/error_log.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                f.write(str(err))
                traceback.print_exc(file=f)
            # penalty
            fitness += 100

        print("fitness: %.2f"%fitness)

    return (fitness,)




"""
The Evolutionary Algorithm
"""

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# initialize a random reservoir
def initIndividual(ICreator):
    sparsity = random.uniform(0,1)
    weights = []
    for i in range(D_RESERVOIR*D_RESERVOIR):
        p = random.uniform(0,1)
        if p > sparsity:
            weights.append(random.uniform(-1,1))
        else:
            weights.append(0)

    return ICreator(weights)

# initialize random MLP weights
def initIndividual_MLP(ICreator):
    return ICreator([random.uniform(-1,1) for i in range((D_IN_MLP+1)*D_H_MLP+(D_H_MLP+1)*D_OUT_MLP)])

# initialize random readout weights
def initIndividual_w_out(ICreator):
    # the output weights of the trained ESN resulting from the first evolution of reservoir weights have mean=0, std=90
    # initialize output weights by drawing from a similar normal distribution
    return ICreator([numpy.random.normal(0,90) for i in range(D_OUT*(D_RESERVOIR+D_IN))])


def createToolbox():

    toolbox = base.Toolbox()

    toolbox.register("individual", initIndividual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutGaussian, mu=0, sigma=0.3, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=10)
    toolbox.register("evaluate", evaluate)

    return toolbox

def createToolbox_w_out():

    toolbox = base.Toolbox()

    toolbox.register("individual", initIndividual_w_out, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=10)
    toolbox.register("evaluate", evaluate_w_out)

    return toolbox

def createToolbox_MLP():

    toolbox = base.Toolbox()

    toolbox.register("individual", initIndividual_MLP, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5) # ref: Eiben Book
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("evaluate", evaluate_MLP)

    return toolbox


##############################################################
# main function of the EA for evolving reservoir weights
##############################################################
def main():

    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    toolbox = createToolbox()

    """
    EA parameters
    """
    pop_size = 20
    population = toolbox.population(n=pop_size)
    mu = pop_size # number of individuals to select for next generation
    lam = 80 # number of offspring
    cxpb = 0.2 # crossover probability
    mutpb = 0.75 # mutation probability
    ngen = 30 # number of generations

    """
    use an Evolutionary Strategy with (mu,lambda)
    """
    eaMuCommaLambda(population, toolbox, mu, lam, cxpb, mutpb, ngen, stats=stats, halloffame=hof,verbose=True)

    return hof, stats

################################################################
# main function of the EA for evolving readout weights
################################################################
def main_w_out():

    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    toolbox = createToolbox_w_out()

    """
    EA parameters
    """
    pop_size = 20
    population = toolbox.population(n=pop_size)
    mu = pop_size # number of individuals to select for next generation
    lam = 80 # number of offspring
    cxpb = 0.5 # crossover probability
    mutpb = 0.5 # mutation probability
    ngen = 20 # number of generations

    """
    use an Evolutionary Strategy with (mu,lambda)
    """
    eaMuCommaLambda(population, toolbox, mu, lam, cxpb, mutpb, ngen, stats=stats, halloffame=hof,verbose=True)

    return hof, stats


################################################################
# main function of the EA for evolving MLP weights
################################################################
def main_MLP():

    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    toolbox = createToolbox_MLP()

    """
    EA parameters
    """
    pop_size = 20
    population = toolbox.population(n=pop_size)
    mu = pop_size # number of individuals to select for next generation
    lam = 50 # number of offspring
    cxpb = 0.0 # crossover probability
    mutpb = 0.8 # mutation probability
    ngen = 20 # number of generations


    """
    use an Evolutionary Strategy with (mu,lambda)
    """
    eaMuCommaLambda(population, toolbox, mu, lam, cxpb, mutpb, ngen, stats=stats, halloffame=hof,verbose=True)

    return hof, stats


"""
The (mu,lambda) Evolutionary Strategy
This is the 'eaMuCommaLambda'-algorithm provided by DEAP,
extended by some additional code that saves intermediate results
"""
def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    t_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    logbook_filename = './EA_output/logbook_'+t_start+'.txt'

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
        with open(logbook_filename,'w') as f:
            print(logbook, file=f)
        with open('./EA_output/hall_of_fame'+t_start+'.txt','a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' best individuals of EA run:\n')
            i = 1
            for ind in halloffame:
                f.write(str(ind)+'\n')
                if evolve_reservoir:
                    w_reservoir = numpy.asarray(ind).reshape((D_RESERVOIR,D_RESERVOIR))
                    numpy.save('./EA_output/best_reservoir_weights_'+str(i)+'_'+t_start+'.npy',w_reservoir)
                elif evolve_mlp:
                    w_1 = numpy.asarray(ind[:(D_IN_MLP+1)*D_H_MLP]).reshape((D_IN_MLP+1,D_H_MLP))
                    w_2 = numpy.asarray(ind[(D_IN_MLP+1)*D_H_MLP:]).reshape((D_H_MLP+1,D_OUT_MLP))
                    mlp = neuralNet.MLP(D_IN_MLP,D_H_MLP,D_OUT_MLP,w_1,w_2)
                    mlp.save('./EA_output/best_mlp_'+str(i)+'_'+t_start+'.pkl')
                else:
                    w_out = numpy.asarray(ind).reshape((D_OUT,D_RESERVOIR+D_IN))
                    numpy.save('./EA_output/best_output_weights_'+str(i)+'_'+t_start+'.npy',w_out)
                i += 1

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
            with open(logbook_filename,'w') as f:
                print(logbook, file=f)
            with open('./EA_output/hall_of_fame'+t_start+'.txt','a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' best individuals of EA run:\n')
                i = 1
                for ind in halloffame:
                    #f.write(str(ind)+'\n')
                    if evolve_reservoir:
                        w_reservoir = numpy.asarray(ind).reshape((D_RESERVOIR,D_RESERVOIR))
                        numpy.save('best_reservoir_weights_'+str(i)+'_'+t_start+'.npy',w_reservoir)
                    elif evolve_mlp:
                        w_1 = numpy.asarray(ind[:(D_IN_MLP+1)*D_H_MLP]).reshape((D_IN_MLP+1,D_H_MLP))
                        w_2 = numpy.asarray(ind[(D_IN_MLP+1)*D_H_MLP:]).reshape((D_H_MLP+1,D_OUT_MLP))
                        mlp = neuralNet.MLP(D_IN_MLP,D_H_MLP,D_OUT_MLP,w_1,w_2)
                        mlp.save('./EA_output/best_mlp_'+str(i)+'_'+t_start+'.pkl')
                    else:
                        w_out = numpy.asarray(ind).reshape((D_OUT,D_RESERVOIR+D_IN))
                        numpy.save('./EA_output/best_output_weights_'+str(i)+'_'+t_start+'.npy',w_out)
                    i += 1
                print(halloffame, file=f)
        with open('./EA_output/logbook_'+t_start+'.pkl','wb') as f:
            pickle.dump(logbook, f)
        with open('./EA_output/population_'+t_start+'.pkl','wb') as f:
            pickle.dump(population, f)

    return population, logbook

"""
Gaussian mutation
This is the 'mutGaussian'-function provided by DEAP,
it is slightly adjusted so that only non-zero weights are perturbed, preventing the mutation to change the reservoir structure
"""
def mutGaussian(individual, mu, sigma, indpb):

    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            if individual[i] != 0:
                individual[i] += random.gauss(m, s)

    return individual,




##########################
# run evolution
##########################
t_start=''

for i in range(1):
    try:
        hof, stats = main()
    except Exception as err:
        # write error info to file
        with open('./error_log.txt', 'a') as f:
            f.write('-------'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'-------------\n')
            f.write(str(err))
            traceback.print_exc(file=f)
        print('-----------terminated with error!-----------------')


########################################
# evaluate best individual once again
########################################
"""
weights = []
with open('offline_evolution_2/best10_after_gen15/best_reservoir_weights_1_2017-12-03_19-29-26.npy', 'rb') as f:
    weights = numpy.load(f)
print(evaluate(weights))
"""


#############################################################
# evaluate ESN with randomly initialized reservoir weights
#############################################################
"""
print(evaluate_net('trained_nn/esn.pkl'))
"""
