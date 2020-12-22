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
from tools.read_write_voxelyze_2d import write_voxelyze_file, read_voxlyze_results


def run_robot_simulation(params, details =False, save_name=""):

  #Caliculation initial cell states
  individual_id = params[0][0]
  w = params[0][1]
  generations = params[1]
  settings = params[2]
  run_directory = settings["run_directory"]
  im_size = settings["im_size"]
  num_classes = settings["voxel_types"]
  input_dim = settings["number_neighbors"]

  if settings['growth_facter']:
    input_dim = 9*2
    voxel_types = settings['voxel_types'] + 1

  if settings['cell_sleep']:
    input_dim = input_dim + 1
    num_classes = num_classes + 1 
  #print(input_dim, num_classes)
  #Make .vxa file for voxelyze

  # Setting up the simulation object
  sim = Sim(dt_frac= settings['fraction'], simulation_time=settings['simtime'], fitness_eval_init_time=settings['initime'])
  # Setting up the environment object
  env = Env(sticky_floor=0, time_between_traces=0)

  p = MorphNet(input_dim = input_dim, number_state = num_classes,  recurrent = settings['recurrent'])

  vector_to_parameters( torch.tensor (w,dtype=torch.float32 ),  p.parameters() )

  morphogens = np.zeros(shape=(1, im_size, im_size))
  morphogens[0, (im_size-1)/2, (im_size-1)/2] = 1
  
  if settings['growth_facter']:
    morphogens = np.zeros(shape=(2, im_size, im_size))
    morphogens[:, (im_size-1)/2, (im_size-1)/2] = 1
  #print(morphogens.shape)

  morphogens_sleep = np.zeros(shape=(1, im_size, im_size))
  out_sleep = torch.zeros(1, im_size*im_size, num_classes-1)
  out = torch.zeros(1, im_size*im_size, num_classes)
  #print(out.shape, out[0].shape, out[0][0], out[0][counter][input_dim-1])
  
  dev_states =[]

  sleep_id = []

  iterations = 10

  hidden_dim = 64

  #Store seperate LSTM hidden states for each cell
  hidden_states = torch.ones(size=(im_size, im_size, 2, 1, im_size*im_size, hidden_dim)) #batch_size = im_size*im_size
  hidden_states_batched_A = torch.ones(size=(1, im_size*im_size, hidden_dim))   #Hidden layers, batch_size, number units
  hidden_states_batched_B = torch.ones(size=(1, im_size*im_size, hidden_dim))   #Hidden layers, batch_size, number units

  #hidden_states = torch.ones(size=(im_size, im_size, 2, 1, 1, hidden_dim)) #batch_size = im_size*im_size
  morph_tmp = copy.deepcopy(morphogens)

  #Insialize a fitness
  max_fitness = 0
  #Start main loop
  for it in range(iterations):
    fitness=0
    batch_inputs = torch.zeros(im_size*im_size, input_dim)
    counter = 0
    sleep_id = []

    if settings['cell_sleep']:
      for i in range(1,im_size-1):
        for e in range(1,im_size-1):
        
          if out[0][counter][num_classes-1] <= 0 and counter not in sleep_id:
            morphogens_sleep[0, i, e] = morphogens[0, i, e]
          else:
            #print(out[0][counter][input_dim-1])
            morphogens_sleep[0, i, e] = 0
    
    #print(batch_inputs[counter].shape,batch_inputs[counter][4],input_dim-1)
    for i in range(1,im_size-1):
      for e in range(1,im_size-1):
        #batch_inputs[counter] = torch.tensor( [morphogens[0, i-1, e],morphogens[0, i+1, e],morphogens[0, i, e+1],morphogens[0, i, e-1],morphogens[0,i,e] ] )            
        
        if settings['cell_sleep']:
          if out[0][counter][num_classes-1] <= 0  and counter not in sleep_id:
             cell_input = torch.tensor( [morphogens_sleep[0, i-1, e], morphogens_sleep[0, i+1, e], morphogens_sleep[0, i, e+1], morphogens_sleep[0, i, e-1], morphogens_sleep[0,i,e], out[0][counter][num_classes-1] ] ) 
          else:
             cell_input = torch.zeros(input_dim)
        elif settings['growth_facter']:
          cell_input = torch.flatten (torch.tensor( morphogens[:, i-1:i+2, e-1:e+2]  ) )

        else: 
          cell_input = torch.tensor( [morphogens[0, i-1, e],morphogens[0, i+1, e],morphogens[0, i, e+1],morphogens[0, i, e-1],morphogens[0,i,e] ] )          
        
        batch_inputs[counter] = cell_input
        #print(input_dim)
        #print(out[0][counter])
        #print(counter)
        counter += 1
      
    output, hs = p(batch_inputs, (hidden_states_batched_A, hidden_states_batched_B)  )
    #print(output.shape)
    if not settings['recurrent']: #ALPHA
        output = output.unsqueeze(0) #So we have recurrent one and non-recurrent in same format
        #print(output.shape)
    #print(output[0, counter].shape, output[0, counter])
    #print(out.shape, out[0].shape, output.shape, out_sleep.shape)
    #print(morphogens[0])

    # Map out to cell state(0,1,2,3,4)
    counter = 0 
    for i in range(1,im_size-1):
      for e in range(1,im_size-1):
        #print(i,e)
        #print(i, e, counter, int(counter/(im_size-2))+1, counter%(im_size-2)+1)
        #print (counter, i,e,  morphogens[0, i, e])
        if settings['growth_facter']:
          cell_alive = (np.amax(morphogens[1, i-1:i+2, e-1:e+2]))>0.1 #ALPHA 

          if cell_alive: #ALPHA 

            _, idx = output[0, counter].data[:4].max(0)   #ALPHA

            #print(out[0].shape, out[0, counter].shape,out[0, counter].data.shape)
            #print(out[0, counter].data[:4])
            alpha =output[0, counter].data[4]  #ALPHA
            

            morph_tmp[0, i, e] =int( idx.data.numpy() )
            morph_tmp[1, i, e] = F.relu(alpha) #ALPHA   TODO apply function earlier, should be faster

            if settings['recurrent']:
              hidden_states_batched_A[0, counter] = hs[0][0, counter]
              hidden_states_batched_B[0, counter] = hs[1][0, counter]
              out = output
            out[0] = output

          else:  #ALPHA
            morph_tmp[:, i, e] = 0 #If cell is not alive, set state to 0

            if settings['recurrent']:
              hidden_states_batched_A[0, counter] = hs[0][0, counter]*0.0
              hidden_states_batched_B[0, counter] = hs[1][0, counter]*0.0
          #print(counter)
          counter += 1
          
        else:
          if settings['cell_sleep']:
            #Make cell sleep version morphogen
            if  out[0][counter][num_classes-1] <= 0  and counter not in sleep_id:
              sum_cells = morphogens_sleep[0, i-1, e]+morphogens_sleep[0, i+1, e]+morphogens_sleep[0, i, e+1]+morphogens_sleep[0, i, e-1]+morphogens_sleep[0,i,e]
            else:
              sum_cells = 0
              sleep_id.append(counter)
              sleep_id= list(set(sleep_id))
              #print(sleep_id)    
          else:
            sum_cells = morphogens[0, i-1, e]+morphogens[0, i+1, e]+morphogens[0, i, e+1]+morphogens[0, i, e-1]+morphogens[0,i,e]
          #print(counter, output[counter], out_sleep[0,counter])
          #print(out.shape,out[0,7],output[0, counter][0:1])
          if sum_cells>0: #If any of the surrounded cells or the cell itself is activate, modify the that position
            #print(out[0, counter].data)
            #print(out_sleep[0,1].shape, output[0,1].shape)
            #print(counter, out_sleep[counter])
            # Chose the highest value state
            if settings['cell_sleep']:
              #print(counter, out_sleep[0, counter] )
              if settings['recurrent']:
                out_sleep[0, counter] = output[0, counter][0:num_classes-1]
                #print(out_sleep[0, counter])
                _, idx = out_sleep[0, counter].data.max(0)
              
              else:
                #print(out_sleep.shape, output.shape)
                out_sleep[0, counter] = output[counter][0:num_classes-1]
                _, idx = out_sleep[0, counter].data.max(0)
            else:
              if settings['recurrent']:
                _, idx = output[0, counter].data.max(0)
                #print(_, idx)
              else:
                _, idx = output[counter].data.max(0)

            # Next cell states
            morph_tmp[0, i, e] =int(idx.data.numpy())
          #print(out.shape, out_sleep.shape, output.shape)

          #TODO if it's mot recurrent, "expression" would also not work right now?
          #A is hidden state, B is cell state
          # Update cell state and hideen states
            if settings['recurrent']:
              hidden_states_batched_A[0, counter] = hs[0][0, counter]
              hidden_states_batched_B[0, counter] = hs[1][0, counter]
              out = output
            out[0] = output
          #print(out[0].shape, output.shape)
          #print(counter)
          counter += 1
      #Caliculation each iterartion cell state
    morpho= morphogens[0][np.newaxis, :, :]
    alpha = morphogens[1][np.newaxis, :, :]

    dev_states.append(morpho)
    #print(type(dev_states),len(dev_states))
    morphogens = copy.deepcopy(morph_tmp)
    #print(morphogens.shape)
 
    #print(morpho.shape)
    
    #print(type(morphogens),morphogens.shape)# morphogen[batchsize,ind_id,im_size]
  #print(individual_id, morphogens,morphogens[0][4], morphogens[0][4][5])

  #Save vxa file and caliculate distance
  #print(fitness)
  write_voxelyze_file(sim, env, generations, individual_id, morpho, fitness, settings['im_size'], settings['run_directory'], settings['run_name'])

  #Evaluate with voxelyze
  #start_time = time.time()
  filepath = settings['run_directory'] + "/tempFiles/softbotsOutput--gene_{0}_id_{1}.xml".format( generations, individual_id)  
  #print(filepath)

  if np.any(morpho != 0):
    proc = sub.Popen("./voxelyze  -f " + settings['run_directory'] + "/voxelyzeFiles/" + settings['run_name'] +"--gene_{0}_id_{1}_fitness_0.vxa".format( generations, individual_id), stdout=sub.PIPE, shell=True)
    proc.wait()
    while os.path.exists(filepath) == False:
      sleep(0.0000000001)
  
  else:
    print("gene_{0}_id_{1} is dead".format( generations, individual_id))

  #Read results
  #All subprocess done
  
  #print(filepath)
  if os.path.exists(filepath):
    fitness = read_voxlyze_results(generations, individual_id, filepath)
    #print(fitness)
    sub.Popen("rm " + settings['run_directory'] + "/tempFiles/softbotsOutput--gene_{0}_id_{1}.xml".format( generations, individual_id)  , stdout=sub.PIPE, shell=True)
  else:
    fitness = 0
  sub.Popen("rm " + settings['run_directory'] + "/voxelyzeFiles/" + settings['run_name'] +"--gene_{0}_id_{1}_fitness_0.vxa".format( generations, individual_id), stdout=sub.PIPE, shell=True)
  print("gene_{0}_id_{1}_fitness{2}".format( generations, individual_id,fitness))
  '''
  total_log = [generations, individual_id,fitness]
  str_ = str(total_log)
  with open("{0}/fitnessFiles/gene_{1}_id_{2}_fitness{3}.txt".format(settings['run_directory'],generations, individual_id,fitness), 'wt') as f:
    f.write(str_)
  '''
  #Return fitness
  return fitness, sim, env, generations, individual_id, morpho, hidden_states_batched_A, hidden_states_batched_B, dev_states