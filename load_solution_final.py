################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
from math import fabs,sqrt
from numpy import genfromtxt
import numpy as np
import random
import sys, os
import time
import glob
import datetime

sys.path.insert(0, 'evoman') 
from environment import Environment 

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
from demo_controller import player_controller


def evaluate_population(population, enemies):
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

    #empty lists for keeping values
    fitness     = []
    player_life = []
    enemy_life  = []
    time        = []
    

    #don't use visuals to make experiments faster
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    #setup game environment
    env = Environment(experiment_name=experiment_name,
                playermode="ai",
                player_controller=player_controller(10),
                multiplemode = multiplemode,
                speed="fastest",
                enemymode="static",
                level=2,
                enemies = enemies)

    #get fitness for each individual
    for i in population: 

        f,p,e,t = env.play(pcont=i[0:-1])

        #add fitness value of i-th individual to fitness
        fitness.append(f)
        player_life.append(p)
        enemy_life.append(e)
        time.append(t)
    
    return fitness, player_life, enemy_life, time

def save_data(data, filepath): 
    #save     
    np.savetxt(filepath, 
               data,
               delimiter =", ", 
               fmt ='% s')

def save_solution(population, fitness, filepath):
    #sort solutions by fitness and pick best one
    solutions        = zip(population,fitness)
    solutions_sorted = sorted(solutions, key = lambda x: x[1])
    solution_best    = solutions_sorted[-1][0]

    #save the solution with the highest fitness
    np.savetxt(filepath, 
               solution_best,
               delimiter =", ", 
               fmt ='% s')

def load_solution(filepath):
    """
    loads a solution into a nparray

    Parameters
    ----------
    filepath : string
        path to the csv

    Returns
    -------
    solution : nparray
         a nparray containing weights for a neural net
    """
    solution = genfromtxt(filepath , delimiter=",")

    return solution

def boxplot_data(folder_name, enemies):
    """
    This function evaluates all solutions in a folder on a certain group of enemies. 
    It produces a CSV with fitness for each of these solutions on the given group of enemies.
    Evaluation is repeated 5 times for each solution.

    Parameters
    ----------
    foldername : string
        the solution folder to load
    enemies : list
        list of integers representing what bosses to play against
    """
    data            = ["solution","trial", "fitness"]
    run             = 0
    directory       = "results/" + folder_name +"/solutions"

    #load all solutions from folder
    for filename in os.listdir(directory):
        #list for storing solution
        solutions = []

        #get full filepath
        file = os.path.join(directory, filename)

        #load solution from file
        solution = load_solution(file)

        #evaluate_population expects a list
        solutions.append(solution)

        #evaluate solution 5 times
        for i in range(5):
            #get fitness of solution
            fitness, p,e,t = evaluate_population(solutions, enemies)

            #add fitness as new row to data
            current_data = [run, i, fitness[0]]
            data.append(current_data)

        #increment run by 1
        run += 1

    #save data
    result_path = "results/" + folder_name + "/BoxplotData.csv"
    save_data(data, result_path)

def individual_gain(folder_name, best_solution, enemies):
    """
    This function evaluates a specific solution against a set of enemies and saves the individual gain for each enemy

    Parameters
    ----------
    foldername : string
        the solution folder to load
    best_solution : int
        best solution to load
    enemies : list
        list of integers representing what bosses to play against
    """

    #initialize head of data
    data = ["enemy","player_life", "enemy_life", "individual_gain"]

    #list to store solution
    solutions = []

    directory = "results/" + folder_name +"/solutions/" 
    file      = directory  + str(best_solution) + ".csv" 

    #load the specific solution
    solution = load_solution(file)

    #evaluate population expects a list of solutions
    solutions.append(solution)

    #evaluate solution against every enemy 5 times
    for e in enemies:

        #evaluate population expects a list of enemies
        enemy = []
        enemy.append(e)

        for i in range(5):
            #use player_life and enemy_life
            f, player_life, enemy_life, t = evaluate_population(solutions, enemy)

            #calculate individual gain
            individual_gain = player_life[0] - enemy_life[0]

            #add new row to the data
            current_data = [e, player_life[0], enemy_life[0], individual_gain ]
            data.append(current_data)

    #save data
    result_path = "results/" + folder_name + "/IndividualGain" + str(best_solution) + ".csv"
    save_data(data, result_path)



#enemies to evaluate against (for task 2 this should be all enemies)
enemies         = [1,2,3,4,5,6,7,8] 

#foldername of the solutions to load 
folder_name     = "enemies[1, 3, 7]SelfSigma_no_2022-10-12 16:32:33.227843"

#best solution given by highest fitness in boxplotdata
best_solution   = 7 

#produce boxplotdata for all solutions
boxplot_data(folder_name, enemies)

#produce individual gain for best individual
# individual_gain(folder_name, best_solution, enemies)