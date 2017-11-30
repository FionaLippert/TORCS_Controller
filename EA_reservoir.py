#!/usr/bin/python3
import random
from deap import base,creator,tools,algorithms,cma
import numpy
import simulate_and_evaluate
import train_ESN
import traceback
import datetime

"""
Evolutionary Algorithm
"""

D_RESERVOIR = 50


def evaluate(individual):

    # convert list of weights to a square matrix
    w_reservoir = numpy.asarray(individual).reshape((D_RESERVOIR,D_RESERVOIR))

    # training
    try:
        train_ESN.train2(w_reservoir)
    except Exception as err:
        # write error info to file
        with open('./error_log.txt', 'a') as f:
            f.write('-------'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'-------------\n')
            f.write(str(err))
            traceback.print_exc(file=f)
    """
    # simulate race on 3 different tracks
    dist = 0
    for i in range(3):
        try:
            simulate_and_evaluate.simulate(i)
            d = simulate_and_evaluate.get_distance_after_time(30.0)
        except Exception as err:
            # write error info to file
            with open('./error_log.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                f.write(str(err))
                traceback.print_exc(file=f)
            d = -1.0 # penalty
        dist += d
        print("distance raced: %.2f"%d)
    """

    try:
        simulate_and_evaluate.simulate()
        dist = simulate_and_evaluate.get_distance_after_time(60.0)
    except Exception as err:
        # write error info to file
        with open('./error_log.txt', 'a') as f:
            f.write('-------'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'-------------\n')
            f.write(str(err))
            traceback.print_exc(file=f)
        dist = -1.0 # penalty

    print("distance raced: %.2f"%dist)
    return (-dist,)


"""
Evolutionary Strategies with (mu,lambda)
"""

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)

def initIndividual(ICreator):
    return ICreator([random.uniform(0,1)*2.0-1.0 for i in range(D_RESERVOIR*D_RESERVOIR)])

toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5) # ref: Eiben Book
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)

hof = tools.HallOfFame(3)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)


def main():

    """
    numpy.random.seed(128)

    strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=2*N)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    algorithms.eaGenerateUpdate(toolbox, ngen=2, stats=stats, halloffame=hof)
    """

    pop_size = 20
    population = toolbox.population(n=pop_size)
    mu = pop_size # number of individuals to select for next generation
    lam = 50 # number of offspring
    cxpb = 0.2 # crossover probability
    mutpb = 0.8 # mutation probability
    ngen = 20 # number of generations


    algorithms.eaMuCommaLambda(population, toolbox, mu, lam, cxpb, mutpb, ngen, stats=stats, halloffame=hof)

    return hof, stats

try:
    hof, stats = main()
except Exception as err:
    # write error info to file
    with open('./error_log.txt', 'a') as f:
        f.write('-------'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'-------------\n')
        f.write(str(err))
        traceback.print_exc(file=f)
    print('-----------terminated with error!-----------------')


with open('./hall_of_fame.txt','a') as f:
    f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' best individuals of EA run:\n')
    for ind in hof:
        f.write(str(ind)+'\n')
with open('./statistics.txt','a') as f:
    f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' statistics of evolution:\n')
    f.write(str(stats))
#final_pop = main()
#print(final_pop)
