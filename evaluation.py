import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import torch.nn.functional as F
import numpy as np
import copy
import subprocess as sub
import os
from time import sleep
from base import Sim, Env
from genotype_net import MorphNet
from tools.read_write_voxelyze import write_voxelyze_file, read_voxlyze_results

def run_robot_simulation(params, details =False, save_name=""):

  #Caliculation initial cell states
  individual_id = params[0][0]
  w = params[0][1]
  generations = params[1] #4
  settings = params[2] #5
  run_directory = settings["run_directory"]
  im_size = settings["im_size"]
  num_classes = settings["voxel_types"]
  input_dim = settings["number_neighbors"]
  
  if settings['growth_facter']:
    input_dim = 3*9*2
    num_classes = settings['voxel_types'] + 1
  
  #Make .vxa file for voxelyze
  # Setting up the simulation object
  sim = Sim(dt_frac= settings['fraction'], simulation_time=settings['simtime'], fitness_eval_init_time=settings['initime'])
  # Setting up the environment object
  env = Env(sticky_floor=0, time_between_traces=0)
  
  p = MorphNet(input_dim = input_dim, number_state = num_classes, recurrent = settings['recurrent'])

  vector_to_parameters( torch.tensor (w,dtype=torch.float32 ),  p.parameters() )

  if settings['growth_facter']:
    morphogens = np.zeros(shape=(2, im_size, im_size, im_size))
    morphogens[:, (im_size-1)/2, (im_size-1)/2, (im_size-1)/2] = 1
    cutedgemorphogens = np.zeros(shape=(2, im_size-2, im_size-2, im_size-2))
  else:
    cutedgemorphogens = np.zeros(shape=(2, im_size-2, im_size-2, im_size-2))
    morphogens = np.zeros(shape=(1, im_size, im_size, im_size)) 
  out = torch.zeros(1, im_size*im_size*im_size, num_classes)

  dev_states =[]
  alphalist = []

  sleep_id = []
  iterations = 10
  hidden_dim = 64

  #Store seperate LSTM hidden states for each cell
  if settings['data_read']:
    #morphogens = params[3]
    #hidden_states_batched_A = params[4]
    #hidden_states_batched_B = params[5]
    hidden_states = torch.ones(size=(im_size, im_size, 2, 1, im_size*im_size*im_size, hidden_dim)) #batch_size = im_size*im_size
    hidden_states_batched_A = torch.ones(size=(1, im_size*im_size*im_size, hidden_dim))   #Hidden layers, batch_size, number units
    hidden_states_batched_B = torch.ones(size=(1, im_size*im_size*im_size, hidden_dim))
    max_fitness = params[6]

  else:
    hidden_states = torch.ones(size=(im_size, im_size, 2, 1, im_size*im_size*im_size, hidden_dim)) #batch_size = im_size*im_size
    hidden_states_batched_A = torch.ones(size=(1, im_size*im_size*im_size, hidden_dim))   #Hidden layers, batch_size, number units
    hidden_states_batched_B = torch.ones(size=(1, im_size*im_size*im_size, hidden_dim))
    max_fitness = 0

  #hidden_states = torch.ones(size=(im_size, im_size, 2, 1, 1, hidden_dim)) #batch_size = im_size*im_size
  morph_tmp = copy.deepcopy(morphogens)

  #Insialize a fitness
  #Start main loop
  for it in range(iterations):
    fitness=0
  
    batch_inputs = torch.zeros(im_size*im_size*im_size, input_dim)

    counter = 0
    sleep_id = []

    for i in range(1,im_size-1):
      for j in range(1,im_size-1):
        for k in range(1,im_size-1):
          if settings['growth_facter']:
            cell_input = torch.flatten (torch.tensor( morphogens[:, i-1:i+2, j-1:j+2, k-1:k+2]  ) )
          else: 
            cell_input = torch.tensor( [morphogens[0, i-1, j, k],morphogens[0, i+1, j, k],morphogens[0, i, j+1, k],morphogens[0, i, j-1, k],morphogens[0,i,j,k-1],morphogens[0,i,j,k+1],morphogens[0,i,j,k] ] )
          batch_inputs[counter] = cell_input
          counter += 1

    output, hs = p(batch_inputs, (hidden_states_batched_A, hidden_states_batched_B) ) 
    past_hidden_states_batched_A = hidden_states_batched_A
    past_hidden_states_batched_B = hidden_states_batched_B

    if not settings['recurrent']: #ALPHA
      output = output.unsqueeze(0) #
    # Map out to cell state(0,1,2,3,4)
  
    counter = 0 
    for i in range(1,im_size-1):
      for j in range(1,im_size-1):
        for k in range(1,im_size-1):

          cell_alive = (np.amax(morphogens[1, i-1:i+2, j-1:j+2, k-1:k+2 ]))>0.1 #ALPHA 

          if cell_alive: #ALPHA 

            _, idx = output[0, counter].data[:5].max(0)   #ALPHA

            alpha =output[0, counter].data[5]  #ALPHA

            morph_tmp[0, i, j, k] =int( idx.data.numpy() )
            morph_tmp[1, i, j, k] = F.sigmoid(alpha) #ALPHA   TODO apply function earlier, should be faster
            if settings['recurrent']:
              hidden_states_batched_A[0, counter] = hs[0][0, counter]
              hidden_states_batched_B[0, counter] = hs[1][0, counter]
              out = output
            out[0] = output

          else:  #ALPHA
            morph_tmp[:, i, j,k] = 0 #If cell is not alive, set state to 0

            if settings['recurrent']:
              hidden_states_batched_A[0, counter] = hs[0][0, counter]*0.0
              hidden_states_batched_B[0, counter] = hs[1][0, counter]*0.0
    
          counter += 1

    #Caliculation each iterartion cell state
    morpho = morphogens[0][np.newaxis, :, :]
    alpha = morphogens[1][np.newaxis, :, :]
    dev_states.append(morpho)
    alphalist.append(alpha)
    morphogens = copy.deepcopy(morph_tmp)

  #Check connection
  counter = -1
  while counter < 0:
    counter = 0
    for i in range(1,im_size-1):
      for j in range(1,im_size-1):
        for k in range(1,im_size-1):
          if morphogens[0, i, j, k] != 0:
            neighbors = np.array([morphogens[0, i-1, j, k], morphogens[0, i+1, j, k], morphogens[0, i, j-1, k], morphogens[0, i, j+1, k], morphogens[0, i, j, k-1], morphogens[0, i, j, k+1]])
            diagonal = np.array([morphogens[0, i-1, j+1, k-1], morphogens[0, i-1, j+1, k+1],morphogens[0, i-1, j+1, k], morphogens[0, i-1, j, k-1], morphogens[0, i-1, j, k+1], morphogens[0, i-1, j-1, k-1],morphogens[0, i-1, j-1, k+1],morphogens[0, i-1, j-1, k],\
            morphogens[0, i, j+1, k-1], morphogens[0, i, j+1, k+1], morphogens[0, i, j-1, k-1],morphogens[0, i, j-1, k+1],\
            morphogens[0, i+1, j+1, k-1], morphogens[0, i+1, j+1, k+1],morphogens[0, i+1, j+1, k], morphogens[0, i+1, j, k-1], morphogens[0, i+1, j, k+1], morphogens[0, i+1, j-1, k-1],morphogens[0, i+1, j-1, k+1],morphogens[0, i+1, j-1, k]])
            if np.count_nonzero(neighbors) == 0 and np.count_nonzero(diagonal) != 0:
              morphogens[0, i, j, k] = 0
              morphogens[1, i, j, k] = 0
              counter -= 1
    
    for i in range(1,im_size-1):
      for j in range(1,im_size-1):
        for k in range(1,im_size-1):
          if morphogens[0, i, j, k] != 0:
            neighbors = np.array([morphogens[0, i-1, j, k], morphogens[0, i+1, j, k], morphogens[0, i, j-1, k], morphogens[0, i, j+1, k], morphogens[0, i, j, k-1], morphogens[0, i, j, k+1]])
            diagonal = np.array([morphogens[0, i-1, j+1, k-1], morphogens[0, i-1, j+1, k+1],morphogens[0, i-1, j+1, k], morphogens[0, i-1, j, k-1], morphogens[0, i-1, j, k+1], morphogens[0, i-1, j-1, k-1],morphogens[0, i-1, j-1, k+1],morphogens[0, i-1, j-1, k],\
            morphogens[0, i, j+1, k-1], morphogens[0, i, j+1, k+1], morphogens[0, i, j-1, k-1],morphogens[0, i, j-1, k+1],\
            morphogens[0, i+1, j+1, k-1], morphogens[0, i+1, j+1, k+1],morphogens[0, i+1, j+1, k], morphogens[0, i+1, j, k-1], morphogens[0, i+1, j, k+1], morphogens[0, i+1, j-1, k-1],morphogens[0, i+1, j-1, k+1],morphogens[0, i+1, j-1, k]])
            all_neighbors = np.count_nonzero(neighbors) + np.count_nonzero(diagonal) 
            if all_neighbors == 0:
              morphogens[0, i, j, k] = 0
              morphogens[1, i, j, k] = 0
   
  cutedgemorphogens[0] = morphogens[0][1:im_size-1,1:im_size-1,1:im_size-1]
  cutedgemorphogens[1] = morphogens[1][1:im_size-1,1:im_size-1,1:im_size-1]

  #Save vxa file and caliculate distance
  write_voxelyze_file(sim, env, generations, individual_id, cutedgemorphogens, fitness, settings['im_size'], settings['run_directory'], settings['run_name'])
  
  #Evaluate with voxelyze
  #start_time = time.time()
  filepath = settings['run_directory'] + "/tempFiles/softbotsOutput--gene_{0}_id_{1}.xml".format( generations, individual_id)  
  if np.any(morphogens[0] != 0):
    proc = sub.Popen("./voxelyze  -f " + settings['run_directory'] + "/voxelyzeFiles/" + settings['run_name'] +"--gene_{0}_id_{1}_fitness_0.vxa".format( generations, individual_id), stdout=sub.PIPE, shell=True)
    proc.wait()
    while os.path.exists(filepath) == False:
      sleep(0.00000000001)
  else:
    print("gene_{0}_id_{1} is dead".format( generations, individual_id))

  #Read results
  #All subprocess done
  if os.path.exists(filepath):
    # voxels cost
    voxels_cost = np.count_nonzero(cutedgemorphogens[0])*0.0583
    fitness = read_voxlyze_results(generations, individual_id, filepath)-voxels_cost

    sub.Popen("rm " + settings['run_directory'] + "/tempFiles/softbotsOutput--gene_{0}_id_{1}.xml".format( generations, individual_id)  , stdout=sub.PIPE, shell=True)
  else:
    fitness = -20
  sub.Popen("rm " + settings['run_directory'] + "/voxelyzeFiles/" + settings['run_name'] +"--gene_{0}_id_{1}_fitness_0.vxa".format( generations, individual_id), stdout=sub.PIPE, shell=True)

  #Return fitness
  return fitness, sim, env, generations, individual_id, morphogens, hidden_states_batched_A, hidden_states_batched_B, dev_states, alphalist, cutedgemorphogens, out, past_hidden_states_batched_A, past_hidden_states_batched_B
