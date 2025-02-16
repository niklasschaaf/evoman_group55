################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
from math import fabs,sqrt
import numpy as np
import random
import sys, os
import time
import glob
import datetime
import optuna
import logging
import plotly
import concurrent.futures

sys.path.insert(0, 'evoman')
from environment import Environment

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
from demo_controller import player_controller

# mut_step_self   = "yes"     # self adapting mutation step size "yes" or anything
def initialize_population(population_size, genotype_size):
    """
    Initializes population by creating a population_size amount of np arrays
    and filling them with genotype_size random values between 0 and 1
    Parameters
    ----------
    population_size : int
        the amount of individuals to be created
    genotype_size : int
        length of the genotype (equals amount of neurons in hidden layer of network)
    Returns
    -------
    population : nparray
         a list of nparrays
    """

    #variable for storing individuals of population
    population = []

    #add new individual population_size times
    for i in range(population_size):

        #create new individual by initializing nparray of random values (-1;1) with size genotype_size
        individual = np.random.uniform(-1,1, genotype_size)

        #add a sigma to the genotype for seldadaptation.
        individual = np.append(individual, np.random.uniform(0, 1, 1))

        #add new individual to population
        population.append(individual)


    return population

def evaluate_population(population, enemies, hidden_neurons):
    """
    Evaluates a population on fitness by having individuals 'play' against a static enemy
    and returning the fitness for this game
    Parameters
    ----------
    population : list
        list of nparrays which values are weights for a neural net
    enemies : list
        list of integers representing what bosses to play against
    hidden_neurons : int
        integer representing amount of hidden neurons in the controller
    Returns
    -------
    fitness :
         a list of fitness values corresonding to the elements in population
    """

    #determine multiple or single mode
    if len(enemies) > 1:
        multiplemode = "yes"
    else:
        multiplemode = "no"

    #empty list for keeping fitness values
    fitness = []

    #don't use visuals to make experiments faster
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    #setup game environment
    env = Environment(experiment_name=experiment_name,
                playermode="ai",
                player_controller=player_controller(hidden_neurons),
                multiplemode = multiplemode,
                speed="fastest",
                enemymode="static",
                level=2,
                enemies = enemies,
                randomini = "no" )

    #get fitness for each individual
    for i in population:

        f,p,e,t = env.play(pcont=i[0:-1]) # drop the mutation stepsize allele

        #add fitness value of i-th individual to fitness
        fitness.append(f)


    return fitness

def evaluate_individuals(individual):
    """
    New function that returns a list of fitness values of a population same as 'evaluate_population'. Only the process is paralellized.
    ----------
    population : list
        list of genotypes
    Returns
    -------
    fitness :
         a list of fitness values corresonding to the elements in population
    """
    fitness_result = []

    #Multiprocess evaluation function
    with concurrent.futures.ProcessPoolExecutor() as executor:
        fitness_result = executor.map(return_fitness, individual)

    fitness = np.array([i for i in fitness_result])

    return fitness

def return_fitness(individual):
    #returns
    f,p,e,t = env.play(pcont=individual[0:-1])

    return f

def mutate_offspring(offspring, mutation_rate, sigma, mut_type):
    """
    Generate variation in a population by randomly mutating values in the genotype of offspring
    Parameters
    ----------
    offspring : list
        offspring individuals as genotype nparrays
    mutation_rate : float
        the rate (probability) at which to use mutation functions
    sigma : float
        constant (standard deviation) used by mutation operators
    mut_type : string
        type of mutation operator to be use. Only "uniform" or "nuniform"
    Returns
    -------
    offspring : list
         a list of nparrays
    """

    #check what mutation operation to use
    if mut_type == "uniform":

        #go through all offspring
        for i in offspring:

            #go through each value of the genotype
            for j in i:

                #change genotype value based on mutation_rate.
                if random.uniform(0,1) <= mutation_rate:
                    #change the value for a random value
                    j = random.uniform(-1,1)


    #check what mutation operation to use
    if mut_type == "nuniform":

        #go through all offspring
        for i in offspring:

            if mut_step_self == "yes":
                sigma = i[-1]

            #go through each value of the genotype
            for j in i:

                #change genotype value based on mutation_rate.
                if random.uniform(0,1) <= mutation_rate:
                    j += random.gauss(0, sigma)


    return offspring

def combine_parents(parents, cross_rate, alpha, cross_type):
    """
    Generate a new generation of offspring by combining the genotypes of parents in several ways
    Parameters
    ----------
    parents : list
        parent individual genotypes to be used for recombination (!expected to be an even amount)
    cross_rate : float
        the rate (probability) at which to use the crossover functions
    alpha : float
        constant used by crossover operators
    cross_type : string
        type of crossover operator to be use. Only "simple", 'single',  "whole" or "blend"
    Returns
    -------
    offspring : list
         a list of nparrays
    """

    #empty list for offspring to be created
    offspring = []

    #check what type of recombination to use
    if cross_type == "simple":

        #loop through pairs of parents in parents
        for i in range(0, len(parents), 2):

            #only cross parents if cross_rate is met
            if random.uniform(0,1) < cross_rate:

                #pick a random recombination point
                k = random.randint(0, len(parents[i])-1)

                #copy parents to children up to k
                child_1 = parents[i][0:k]
                child_2 = parents[i+1][0:k]

                #empty array for storing combined values
                geno_tail = np.zeros(len(parents[i])- k)

                #calculate the 'tail' of the genotype
                for j in range(len(parents[i]) - k):
                    geno_tail[j] = (alpha * parents[i][j + k] ) + ((1- alpha) * parents[i+1][j +k])

                #add the tail to the genotype of children making them complete individuals
                child_1 = np.concatenate((child_1,geno_tail))
                child_2 = np.concatenate((child_2,geno_tail))

                #add children to offspring
                offspring.append(child_1)
                offspring.append(child_2)

            else:
                #parents are used as offspring
                offspring.append(parents[i])
                offspring.append(parents[i+1])


    #check what type of recombination to use
    if cross_type == "single":

        #loop through pairs of parents in parents
        for i in range(0, len(parents), 2):

            #only cross parents if cross_rate is met
            if random.uniform(0,1) < cross_rate:

                #pick a random recombination point
                k = random.randint(0, len(parents[i])-1)

                #copy parents genotype to children genotype
                child_1 = parents[i]
                child_2 = parents[i+1]

                #combine values at position k
                k_combined = (alpha * parents[i +1][k] ) + ((1- alpha) * parents[i][k])
                child_1[k] = k_combined
                child_2[k] = k_combined

                #add children to offspring
                offspring.append(child_1)
                offspring.append(child_2)

            else:
                #parents are used as offspring
                offspring.append(parents[i])
                offspring.append(parents[i+1])


    #check what type of recombination to use
    if cross_type == "whole":

        #loop through pairs of parents in parents
        for i in range(0, len(parents), 2):

            #only cross parents if cross_rate is met
            if random.uniform(0,1) < cross_rate:

                #genotype with empty values
                child = np.zeros(len(parents[i]))

                #fill genotype according to 'whole' recombination operation
                for j in range(len(parents[i])):

                    #calculate mean of all values in parent pairs
                    child[j] = (alpha * parents[i][j] ) + ((1- alpha) * parents[i+1][j])

                #same child is added twice (because 'whole' creates duplicate children')
                offspring.append(child)
                offspring.append(child)

            else:
                #parents are used as offspring
                offspring.append(parents[i])
                offspring.append(parents[i+1])


    #check what type of recombination to use
    if cross_type == "blend":

        #loop through pairs of parents in parents
        for i in range(0, len(parents), 2):

            #only cross parents if cross_rate is met
            if random.uniform(0,1) < cross_rate:

                #genotype with empty values
                child_1 = np.zeros(len(parents[i]))
                child_2 = np.zeros(len(parents[i]))

                #fill genotypes according to 'blend' recombination operation
                for j in range(len(parents[i])):

                    #sample random number uniformly
                    u = random.uniform(0,1)

                    print("uu1 " + str(u))
                    #calculate gamma
                    gamma = (1 - 2*alpha) * u - alpha

                    print("gamma_1  " + str(gamma))
                    #calculate j-th value of child
                    child_1[j] = (1 - gamma) * parents[i][j] + gamma * parents[i+1][j]

                    #same process for child_2 with new random u
                    u = random.uniform(0,1)
                    print("uu2 " + str(u))
                    gamma = (1 - 2*alpha) * u - alpha
                    print("gamma_2  " + str(gamma))
                    child_2[j] =  (1 - gamma) * parents[i][j] + gamma * parents[i+1][j]


                    print("p1 ")
                    print(parents[i])
                    print("\n\n p2")
                    print(parents[i + 1])
                    print("\n\n ch1")
                    print(child_1)
                    print("\n\n child_2")
                    print(child_2)

                #add children to offspring
                offspring.append(child_1)
                offspring.append(child_2)

            else:
                #parents are used as offspring
                offspring.append(parents[i])
                offspring.append(parents[i+1])

    return offspring

def select_parents(pop_gen, pop_fit, num_parent, sample_size):
    """
    Make a selection of parents from the population using the tournament selection method
    The fittest individual will always win.
    Parameters
    ----------
    pop_gen :list
        genomes of the population
    pop_fit : list
        fitness of population
    pop_par : list
        parents of population
    pop_par : list
        gradparents of population
    num_parent : int
        number of parents to be selected. should be even
    sample_size : int
        number of individuals drawn from population for tournament
    Returns
    -------
    parents : list
         genomes of the parents
    parents_par : list
         genomes of the parents parents
    to add
    ------
    - input list with identifiers for individuals from current population
    - output lists with identifiers for both parents
    """
    parents = [] # list of parents

    # check that number of parents is even
    if not num_parent % 2 == 0:
        num_parent = num_parent+1
        print('number of parents uneven: '+str(num_parent-1)+' corrected to: '+str(num_parent))

    # check that number of parents isn't too large
    if num_parent > len(pop_gen):
        exit('!!!number of requested parents ('+str(num_parent)+') larger than population size ('+str(len(pop_gen))+')!!!')

    # initialize list that can be used to draw a random sample and then serves
    # and then serves as index for pop_gen and pop_fit
    index = []
    for i in range(len(pop_gen)):
        index.append(i)

    # select the parents
    for i in range(num_parent):

        # get parent sample
        sample = []
        sample = random.choices(index, k = sample_size)

        sample_fit = [] # fitness of the sample
        for j in range(sample_size):
            sample_fit.append(pop_fit[sample[j]])

        index_fittest = sample[sample_fit.index(max(sample_fit))] # index of the fittest individual in sample

        # write to first parent list if i is even
        parents.append(pop_gen[index_fittest])

    return(parents)

def select_survivors(pop_gen, pop_fit, child_pop, child_fit, pop_size, sample_size):
    """
    ############################
    # runs in tournament mode and removes the worst individual out of the sample
    # from the combined parent-children population
    """

    surv_pop = [] # genome list of population that survives
    surv_fit = [] # fitness list of population that survives

    purge_pop = []
    purge_fit = []
    purge_pop.extend(pop_gen) # list of parents + children
    purge_pop.extend(child_pop)
    purge_fit.extend(pop_fit) # combined list of fitness
    purge_fit.extend(child_fit)

    # initialize list that can be used to draw a random sample and then serves
    # as index for pop_gen and pop_fit

    # select the parents
    while len(purge_pop) > pop_size:
        index = []
        for i in range(len(purge_pop)): # index length needs to be updated each loop
            index.append(i)

        # get parent sample
        sample = []
        sample = random.choices(index, k = sample_size)

        sample_fit = [] # fitness of the sample
        for j in range(sample_size):
            sample_fit.append(purge_fit[sample[j]])

        index_leastfit = sample[sample_fit.index(min(sample_fit))] # finds index of the least fit individual in sample

        # if not purge_fit[index_leastfit] == min(sample_fit):
        #     print("!!!!!ALARM!!!!!")


        # test_fitness = return_fitness(purge_pop[index_leastfit])
        # if not test_fitness == min(sample_fit):
        #     print(str(test_fitness))
        #     print(str(min(sample_fit)))

        purge_pop.pop(index_leastfit)
        purge_fit.pop(index_leastfit)

    surv_pop.extend(purge_pop)
    surv_fit.extend(purge_fit)

    return(surv_pop, surv_fit)


##CODE FOR EXPERIMENT VARIABLES
# Experiment variables
hidden_neurons  = 10        #number of hidden neurons in the controller (DON'T CHANGE)
total_weights   = (20+1)*hidden_neurons + (hidden_neurons+1)*5 #number ofweights in neural net (DON'T CHANGE)

population_size = 50         #amount of solutions to evolve
cross_rate      = 1       #rate (probability) at which crossover operator is used. if 1 always crossover, if 0 never crossover
alpha           = 0.5        #constant used by crossover operators in combine_parents
mutation_rate   = 0.72       #rate (probability) at which mutations occur (mutate_offspring)
model_runtime   = 30          #number of generations the EA will run
tournament_size = 3           #amount of tournaments done in select_parents and select_survivors
parent_n        = 16          #amount of parents in the tournament pool (can't be larger than populationsize)
mut_type        = "nuniform"  #type of mutation operator, can be uniform or nuniform
cross_type      = "single"    #type of crossover operator, can be single, simple, whole or blend
sigma           = 0.99        #standard deviation used by mutation operator nuniform eg. mutation step size

#change these parameters for you experiment :)
enemies         = [1,3,7]        #list of enemies solutions are evaluated against. max is [1,2,3,4,5,6,7,8]
mut_step_self   = "no"        # self adapting mutation step size "yes" or anything


mode = "n0tuning" # set to "tuning" for tuning with optuna anything else for normal run
trials = 5 # trials that optuna uses

#initialize environment globally so the evaluation function can be multiprocessed.
#determine multiple or single mode
if len(enemies) > 1:
    multiplemode = "yes"
else:
    multiplemode = "no"

#empty list for keeping fitness values
fitness = []

#don't use visuals to make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

#setup game environment
env = Environment(experiment_name=experiment_name,
            playermode="ai",
            player_controller=player_controller(hidden_neurons),
            multiplemode = multiplemode,
            speed="fastest",
            enemymode="static",
            level=2,
            enemies = enemies)

#time counter for testing multiprocessing
start = time.perf_counter()
if __name__ == '__main__': # protect code for multiprocessing
    if mode == "tuning":
        # objective function for optimization with optuna
        def tuning(trial):
            """
            objective function that can be used by optuna to find the best parameter
            right now it optimizes for best fitness gain in average and maximum fitness
            """

            # # fixed experiment variables
            model_runtime   = 10       #number of generations the EA will run
            mut_step_self   = "no"

            # initialize parameters to be optimized
            # mutation_rate = trial.suggest_float('mutation_rate', 0, 1)
            sigma = trial.suggest_float('sigma', 0, 1)
            tournament_size = trial.suggest_int('tournament_size', 2, population_size)
            # parent_n = 2 * trial.suggest_int('half_parent_n', 2, population_size // 2)

            # lists to calculate fitness gain
            avg_fit_first = []
            avg_fit_last = []
            max_fit_first = []
            max_fit_last = []

            #run the entire EA 3 times to achieve more robust results
            for run in range(3):

                ##CODE FOR RUNNING EXPERIMENTS
                #initialize population
                population = initialize_population(population_size, total_weights)

                #determine fitness of entire population generation
                fitness    = evaluate_individuals(population)

                avg_fit_first.append(np.mean(fitness))
                max_fit_first.append(max(fitness))

                ##CODE FOR RUNNING EXPERIMENTS
                #main model loop
                for i in range(model_runtime):
                    #determine parents (does not work as of yet)
                    # print("parent selection")
                    parents = select_parents(population, fitness, parent_n, tournament_size)

                    #cross parents
                    # print("crossover")
                    offspring = combine_parents(parents, cross_rate, alpha, cross_type)

                    #mutate offspring
                    # print("making offspring")
                    offspring = mutate_offspring(offspring, mutation_rate, sigma, mut_type)

                    #determine fitness of offspring
                    # print("determining fitness of offspring")
                    fitness_offspring = evaluate_individuals(offspring)

                    #select survivors
                    # print("select new population")
                    population, fitness = select_survivors(population, fitness, offspring, fitness_offspring, population_size, tournament_size)

                # fitness of last generation to list
                avg_fit_last.append(np.mean(fitness))
                max_fit_last.append(max(fitness))

            # calculate fitness gain
            avg_fit_gain = np.mean(avg_fit_last) - np.mean(avg_fit_first)
            max_fit_gain = np.mean(max_fit_last) - np.mean(max_fit_first)

            return max_fit_gain #avg_fit_gain

        # run the optimization
        #fix_parameters = hidden_neurons, total_weights, population_size, alpha,\
        #model_runtime, mut_type, cross_type, enemies

        # set timestamp
        timestamp        = str(datetime.datetime.now())
        # make directory
        directory = "tuning/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        # start a new optimization
        study_name = "tuning/optuna_"+str(timestamp)  # Unique identifier of the study.

        # # continue exisiting run
        # filename = "optuna_2022-10-11 22:18:23.412403" # without cross_rate, 5 iterations, [3,5,8]
        # study_name = "tuning/"+filename  # Unique identifier of the study.

        storage_name = "sqlite:///{}.db".format(study_name)

        sampler = optuna.samplers.CmaEsSampler() # define sampler
        study = optuna.create_study(directions=["maximize"], sampler = sampler, study_name=study_name, storage=storage_name, load_if_exists=True)
        study.optimize(tuning, n_trials=trials) # n_trial is the number of iterations for the optimization

        # plots
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()

        fig = optuna.visualization.plot_contour(study)
        fig.show()

    else:
        # set timestamp
        timestamp        = str(datetime.datetime.now())

        #run the entire EA 10 times
        for run in range(10):
            print("run: " + str(run))

            ##CODE FOR RUNNING EXPERIMENTS
            #initialize population
            population = initialize_population(population_size, total_weights)

            # #determine fitness of initial population
            fitness    = evaluate_individuals(population)

            #empty list for storing fitness data
            data       = []
            fields = ['Generation', 'Lowest', 'Mean', 'Highest', 'Stdev']
            data.append(fields)

            ##CODE FOR RUNNING EXPERIMENTS
            #main model loop
            for i in range(model_runtime):
                print("generation: " + str(i) + "\n")

                #save fitness of current generation
                fitness_sorted = fitness.copy()
                fitness_sorted.sort()

                current_data = [i, fitness_sorted[0], np.mean(fitness_sorted), fitness_sorted[-1], np.std(fitness_sorted)]
                data.append(current_data)

                #determine parents (does not work as of yet)
                # print("parent selection")
                parents = select_parents(population, fitness, parent_n, tournament_size)

                #cross parents
                # print("crossover")
                offspring = combine_parents(parents, cross_rate, alpha, cross_type)

                #mutate offspring
                # print("making offspring")
                offspring = mutate_offspring(offspring, mutation_rate, sigma, mut_type)

                #determine fitness of offspring
                # print("determining fitness of offspring")
                fitness_offspring = evaluate_individuals(offspring)

                #select survivors
                # print("select new population")
                population, fitness = select_survivors(population, fitness, offspring, fitness_offspring, population_size, tournament_size)


            # save the last generation as well
            fitness_sorted = fitness.copy()
            fitness_sorted.sort()

            current_data = [model_runtime, fitness_sorted[0], np.mean(fitness_sorted), fitness_sorted[-1], np.std(fitness_sorted)]
            data.append(current_data)

            ##CODE FOR SAVING EXPERIMENT DATA
            # make directory
            directory = "results/enemies"+str(enemies[:])+'SelfSigma_'+mut_step_self+'_'+timestamp
            if not os.path.exists(directory):
                os.makedirs(directory+"/fitness/")
                os.makedirs(directory+"/solutions/")

            #experiment names
            experiment_fit   =  directory+"/fitness/"+ str(run) +".csv"
            experiment_sol   =  directory+"/solutions/"+ str(run) +".csv"

            #sort solutions by fitness and pick best one
            solutions        = zip(population,fitness)
            solutions_sorted = sorted(solutions, key = lambda x: x[1])
            solution_best    = solutions_sorted[-1][0]

            #save fitness values of each iteration
            np.savetxt(experiment_fit,
                       data,
                       delimiter =", ",
                       fmt ='% s')

            #save the solution with the highest fitness
            np.savetxt(experiment_sol,
                       solution_best,
                       delimiter =", ",
                       fmt ='% s')


        experiment_par   = directory+"/PARAMETERS.txt"
        #save parameter settings
        with open(experiment_par, 'w') as f:
            f.write("population_size: " + str(population_size))
            f.write("\nhidden_neurons: " + str(hidden_neurons))
            f.write("\ntotal_weights: " + str(total_weights))
            f.write("\ncross_rate: " + str(cross_rate))
            f.write("\nalpha: " + str(alpha))
            f.write("\nmutation_rate: " + str(mutation_rate))
            f.write("\ntournament_size: " + str(tournament_size))
            f.write("\nenemies: " + str(enemies))
            f.write("\nmodel_runtime: " + str(model_runtime))
            f.write("\nparent_n: " + str(parent_n))
            f.write("\nmut_type: " + str(mut_type))
            f.write("\ncross_type: " + str(cross_type))
            f.write("\nsigma: " + str(sigma))

    #print time it took to run the program
    print(f'The program with multiprocessing took {round((time.perf_counter()-start),2)} seconds')
