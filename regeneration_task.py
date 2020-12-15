from __future__ import print_function

import matplotlib
#matplotlib.use('Agg')
import sys
import subprocess as sub
import os
from os.path import exists
import scipy.misc
from matplotlib import colors
import argparse
import numpy as np
import math
import time
import random
import multiprocessing

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)

from GA import GeneticAlgorithm
from genotype_net import MorphNet
from evaluation import regeneration

# Call Voxelyze
#sub.call("cp ~/Dropbox/recurrent_evodevo/archive/2d_creatures/_voxcad/voxelyzeMain/voxelyze .", shell=True)
sub.call("chmod 755 voxelyze", shell=True)

def main(argv):
    start = time.time()
    parser = argparse.ArgumentParser()

  #Creatuer settings
    parser.add_argument('--im_size', type=int, default=9, metavar='N',\
      help='Size of creature')
    parser.add_argument('--voxel_types', type=int, default=5, metavar='N',\
      help='How many different types of voxels')
    parser.add_argument('--number_neighbors', type=int, default=7, metavar='N',\
      help='Number of neighbors')
    parser.add_argument('--initial_noise', type=int, default=1, metavar='N',\
      help='initial_noise')
  # GA setting
    parser.add_argument('--sigma', type=float, default = 0.03,\
      help='Sigma')
    parser.add_argument('--N', type=int, default = 3,\
      help='N')
  # Voxelyze settings
    parser.add_argument('--simtime',type=int,default=0.55,\
      help='Simulation time in voxelyze')
    parser.add_argument('--initime',type=int,default=0.05,\
      help='Intiation time of simulation in voxelyze')
    parser.add_argument('--fraction',type=float,default=0.9,\
      help='Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.')
    parser.add_argument('--run_directory',type=str,default='regeneration')
    parser.add_argument('--run_name',type=str,default='Regeneration')
  #General settings
    parser.add_argument('--popsize', type=int, default =10, \
      help='Population size.') #3000
    parser.add_argument('--generations', type=int, default = 4,\
      help='Generations.') #50000 
    parser.add_argument('--threads', type=int, default=4, metavar='N',\
      help='threads')
    parser.add_argument('--optimizer', type=str, default='ga', metavar='N',\
      help='ga')
    parser.add_argument('--recurrent', type=int, default=0, metavar='N',\
      help='0 = not recurrent, 1 = recurrent')
    parser.add_argument('--data_read', type=int, default=0, metavar='N',\
      help='0 = not data read, 1 = dataread')
    parser.add_argument('--cell_sleep', type=int, default=0, metavar='N',\
      help='0 = not sleep, 0 = sleep')
    parser.add_argument('--growth_facter', type=int, default=1, metavar='N',\
      help='0 = not growt facter, 1 = growth facter exist')
    parser.add_argument('--seed', type=int, default=1, metavar='N',\
      help='seed')
    parser.add_argument('--folder', type=str, default='results', metavar='N',\
      help='folder to store results')
    parser.add_argument('--show', type=str, default='none', metavar='N',\
      help='visualize genome') 
    parser.add_argument('--expression', type=int, default=0, metavar='N',\
      help='seperate gene expression') 
    parser.add_argument('--fig_output_rate', type=int, default=1, metavar='N',\
      help='fig_output_rate') 
    
    args = parser.parse_args()
    
    # You need to add variables that you want to refer to in evaluation.py.
    settings = {'voxel_types':args.voxel_types, 'im_size':args.im_size, 'optimizer':args.optimizer, \
    'recurrent' : args.recurrent, 'cell_sleep': args.cell_sleep, 'growth_facter':args.growth_facter, 'expression': args.expression, 'seed':args.seed, 'fraction':args.fraction,\
    'simtime':args.simtime, 'initime':args.initime, 'run_directory':args.run_directory, 'run_name':args.run_name, 'data_read':args.data_read,\
    'number_neighbors':args.number_neighbors, 'sigma':args.sigma, 'N':args.N, 'fig_output_rate':args.fig_output_rate, 'initial_noise':args.initial_noise}
    #TODO do not alwaus pass the whole target image
    
    input_dim = settings['number_neighbors']
    voxel_types = settings['voxel_types']
    
    if settings['growth_facter']:
      input_dim = 3*9*2
      voxel_types = settings['voxel_types'] + 1

    if settings['cell_sleep']:
      input_dim = settings['number_neighbors'] + 1
      voxel_types = settings['voxel_types'] + 1 
    #print(input_dim, voxel_types)

    model = MorphNet(input_dim =  input_dim, number_state = voxel_types, recurrent=settings['recurrent'])

    if args.show!='none':
        x = torch.load(args.show)
        ca_fitness(x, True, "iterations.png")
        exit()

    print("OPTIMIZER ",args.optimizer)
    
    inputfile_path = args.run_directory + "/voxelyzeFiles/"
    outputfile_path = args.run_directory + "/fitnessFiles/"
    tempfile_path = args.run_directory + "/tempFiles/"
    bestocreatures_path = args.run_directory + "/bestofFiles/"
    time_path = args.run_directory + "/timeFiles/"
    model_path = args.run_directory + "/model_stateFiles/"
    dev_states_path = args.run_directory + "/dev_stateFiles/"
    try:
        if not os.path.exists(inputfile_path):
            os.makedirs(inputfile_path)
    except OSError as err:
        print(err)
    try:
        if not os.path.exists(outputfile_path):
            os.makedirs(outputfile_path)
    except OSError as err:
        print(err)
    try:
        if not os.path.exists(tempfile_path):
            os.makedirs(tempfile_path)
    except OSError as err:
        print(err)
    try:
        if not os.path.exists(bestocreatures_path):
            os.makedirs(bestocreatures_path)
    except OSError as err:
        print(err)
    try:
        if not os.path.exists(time_path):
            os.makedirs(time_path)
    except OSError as err:
        print(err)
    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    except OSError as err:
        print(err)
    try:
        if not os.path.exists(dev_states_path):
            os.makedirs(dev_states_path)
    except OSError as err:
        print(err)
    #0.03 worked for evolving heart
    #print(model.get_weights().shape)
    if args.optimizer=='ga':
      ga = GeneticAlgorithm(model.get_weights(), regeneration, population_size=args.popsize, sigma=0.1, \
        num_threads=args.threads, folder=args.folder, settings=settings) #0.05 works okay
      ga.run(args.generations, print_step=1)
    else:
      print('No optimizer specified')

    process_time = time.time() - start
    str_ = str(process_time)
    with open("{0}/timeFiles/process_time.txt".format( args.run_directory), 'wt') as f:
      f.write(str_)
    #print(process_time, 'sec')
if __name__ == '__main__':
    main(sys.argv)
