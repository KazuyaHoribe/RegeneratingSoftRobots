from __future__ import print_function

import multiprocessing
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib.tri as mtri
import re
from tools.read_write_voxelyze import write_voxelyze_file, read_voxlyze_results, write_voxelyze_file_fitness

LOG = False
FIG = True

if LOG:
    from torch.utils.tensorboard import SummaryWriter

torch.random.seed()

class GeneticAlgorithm():
    def __init__(self, weights, fitness_function, population_size, sigma, num_threads, folder, settings):
        #GeneticAlgorithm.__init__(self, weights, fitness_function, population_size, sigma, num_threads, folder, settings) 
        #! Python 3super(GeneticAlgorithm, self).__init__()
        self.weight_shape = len(weights)
        self.fitness_function = fitness_function
        self.pop_size = population_size
        self.sigma = sigma
        self.num_threads = num_threads
        self.folder = folder
        self.settings = settings
        sigma = settings['sigma']

        np.random.seed(self.settings['seed'])

        filename="run_"+str(self.settings['seed'])
        if self.settings['recurrent']:
            filename+="_recurrent"

        if LOG:
            self.writer = SummaryWriter(folder+"/"+filename+"/"+settings['target'])


    def run(self, generations, print_step):

        population = []
        fitness_log =[]
        fitness_all = []
        plot_log = []
        gen=[]
        maxgenfit=[]
        meanfit = []
        maxfit = []

        N = self.settings['N']#self.pop_size/3 #use top 5 individuals

        elitism = N

        if self.settings['data_read']:
            population = torch.load('run1/model_stateFiles/weights_gen_1.pt')
            morphogens = np.load('run1/model_stateFiles/morphogens_gen_1.npy')
            hidden_states_batched_A = torch.load('run1/model_stateFiles/hidden_states_batched_A_gen_0.pt')
            hidden_states_batched_B = torch.load('run1/model_stateFiles/hidden_states_batched_B_gen_0.pt')

            pattern=r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
            f = open('run1/fitnessFiles/fitness_log_gen_1.txt')
            fitness = f.read()
            f.close()
            fitness=re.findall(pattern,fitness)
            fitness = map(float, fitness)
            fitness = np.array(fitness)
            max_fitness = max(fitness[::-4])

        else:
            for i in range(self.pop_size):
                w = torch.from_numpy( np.random.normal(0, 1, self.weight_shape)*0.1*self.settings['initial_noise']).float()
                population.append( w )
            max_fitness = -20

        pool = multiprocessing.Pool(self.num_threads)

        for epoch in range(0, generations):
            #fitness, sim, env, generations, individual_id, morphogens 

            if self.settings['data_read']:
                results = pool.map(self.fitness_function, [ (it, epoch, self.settings, morphogens[it[0]], hidden_states_batched_A[it[0]], hidden_states_batched_B[it[0]], max_fitness) for it in enumerate(population) ])
            else:
                results = pool.map(self.fitness_function, [ (it, epoch, self.settings) for it in enumerate(population) ])
            # ,hidden_states_batched_A[it],hidden_states_batched_B[it]

            fitness = [row[0] for row in results]
            sim = [row[1] for row in results]
            env = [row[2] for row in results]
            individual_id = [row[4] for row in results]
            morphogens = [row[5] for row in results]
            hidden_states_batched_A = [row[6] for row in results]
            hidden_states_batched_B = [row[7] for row in results]
            dev_states = [row[8] for row in results]
            alpha = [row[9] for row in results]
            cutedgemorphogens = [row[10] for row in results]
            out = [row[11] for row in results]
            past_hidden_states_batched_A = [row[12] for row in results]
            past_hidden_states_batched_B = [row[13] for row in results]

            fitness_all.append(fitness)
            #Get individuales id
            sort_idx = np.argsort([-f for f in fitness])
            max_gen_f = np.max(fitness)

            if (max_gen_f>max_fitness):
                max_fitness = max_gen_f

                write_voxelyze_file_fitness(sim[sort_idx[0]], env[sort_idx[0]], epoch, individual_id[sort_idx[0]], cutedgemorphogens[sort_idx[0]], max_fitness, self.settings['im_size'], self.settings['run_directory'], self.settings['run_name'])
                torch.save(population[sort_idx[0]], "{0}/bestofFiles/weights_gen_{1}_{2}.pt".format(self.settings['run_directory'],epoch,sort_idx[0]))
                np.save('{0}/bestofFiles/morphogens_gen_{1}_id_{2}'.format(self.settings['run_directory'],epoch, sort_idx[0]), morphogens[sort_idx[0]])

                torch.save(past_hidden_states_batched_A[sort_idx[0]],'{0}/bestofFiles/past_hidden_states_batched_A_gen_{1}_id_{2}.pt'.format(self.settings['run_directory'],epoch, sort_idx[0]))
                torch.save(past_hidden_states_batched_B[sort_idx[0]],'{0}/bestofFiles/past_hidden_states_batched_B_gen_{1}_id_{2}.pt'.format(self.settings['run_directory'],epoch, sort_idx[0]))

                dev_states = np.asarray(dev_states)
                alpha = np.asarray(alpha)
                #filename = "{0}/GA_saved_weights_gen_{1}_{2}".format(self.folder,epoch, max_fitness)
                #m = self.fitness_function( (population[sort_idx[0] ], self.settings), True, filename)
                dev_states = dev_states.reshape(len(population),len(dev_states[1]),self.settings['im_size'],self.settings['im_size'],self.settings['im_size'])
                alpha = alpha.reshape(len(population),len(alpha[1]),self.settings['im_size'],self.settings['im_size'],self.settings['im_size'])
                np.save('{0}/bestofFiles/dev_states_gen_{1}_id_{2}'.format(self.settings['run_directory'],epoch, sort_idx[0]), dev_states[sort_idx[0]])
                np.save('{0}/bestofFiles/alpha_gen_{1}_id_{2}'.format(self.settings['run_directory'],epoch, sort_idx[0]), alpha[sort_idx[0]])

                mynorm = plt.Normalize(vmin=0, vmax=1)
                fig = plt.figure(figsize=(20,10))
                for it in range(0,len(dev_states[1])):
                    #print(it)
                    voxels = dev_states[sort_idx[0]][it]
                    voxels = voxels.transpose((2,1,0))
                    alpha_temp = alpha[sort_idx[0]][it]
                    alpha_temp = alpha_temp.transpose((2,1,0))
                    #print(voxels[1],voxels[1,3])
                    
                    ax = fig.add_subplot(2, 5, it+1, projection= '3d')
                    #plt.subplot(3, iterations/3+12, it+1)#,figsize=(15,15))
                    col = [[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0],[0, 1, 0]]
        
                    face_col = np.concatenate( (np.array(col)[voxels.astype(int)], np.expand_dims(alpha_temp, axis=3) ) , axis=3)
                    #face_col = np.concatenate( (np.array(col)[morphocegens[0].astype(int)], np.expand_dims(morphogens[1], axis=3) ) , axis=3) 
                    #face_col = face_col.transpose((1,2,0))
                    ax.set_aspect(aspect=1)
                    ax.voxels( voxels, facecolors=face_col,edgecolor='k')#np.array(col)[morphogens[0].astype(int)])#,

                plt.savefig('{0}/bestofFiles/gen{1}_id{2}.pdf'.format(self.settings['run_directory'],epoch, sort_idx[0]))
                plt.close()

                if LOG:
                    self.writer.add_image("Image", m.transpose(2, 0, 1), epoch)
        
            fitness_log.append((epoch, max_gen_f, np.mean(fitness), max_fitness))
            gen.append (fitness_log[epoch][0])
            maxgenfit.append(fitness_log[epoch][1])
            meanfit.append (fitness_log[epoch][2])
            maxfit.append (fitness_log[epoch][3])
            #print(fitness_log)
            if epoch % self.settings['fig_output_rate'] ==0:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1) 
                #ax.plot(gen, maxgenfit, linestyle='--', color='b', label='maxgenfit')
                ax.plot(gen, meanfit, linestyle='-', color='r', label='Mean')
                ax.plot(gen, maxfit, linestyle='dotted', color = 'b', label='Max fitness')
                ax.set_xlabel('Generations')
                ax.set_ylabel('Fitness')
                ax.legend(loc='best')
                plt.savefig('{0}/epoch{1}.pdf'.format(self.settings['run_directory'],epoch))

                #print(population)
                torch.save(population, "{0}/model_stateFiles/weights_gen_{1}.pt".format(self.settings['run_directory'],epoch))

                str_ = str(fitness_log)
                str_1 = str(fitness_all)
                with open("{0}/fitnessFiles/fitness_log_gen_{1}.txt".format(self.settings['run_directory'], epoch), 'wt') as f:
                    f.write(str_)
                with open("{0}/fitnessFiles/fitness_all_gen_{1}.txt".format(self.settings['run_directory'], epoch), 'wt') as g:
                    g.write(str_1)

                dev_states = np.asarray(dev_states)   
                alpha = np.asarray(alpha)
                #np.save('{0}/dev_stateFiles/dev_states_gen_{1}_id_{2}'.format(self.settings['run_directory'], epoch, sort_idx[0]), dev_states[sort_idx[0]])
                #filename = "{0}/GA_saved_weights_gen_{1}_{2}".format(self.folder,epoch, max_fitness)
                #m = self.fitness_function( (population[sort_idx[0] ], self.settings), True, filename)
                dev_states = dev_states.reshape(len(population),len(dev_states[1]),self.settings['im_size'],self.settings['im_size'],self.settings['im_size'])

                alpha = alpha.reshape(len(population),len(alpha[1]),self.settings['im_size'],self.settings['im_size'],self.settings['im_size'])

            new_pop = []
            for idx in range(self.pop_size-elitism):

                #Select indivdual from the top N
                i = np.random.randint(0, N )
                p = population[ sort_idx[i]]

                new_ind = p + torch.from_numpy( np.random.normal(0, 1, self.weight_shape) * self.sigma).float()

                new_pop.append(new_ind)

            for idx in sort_idx[:elitism]:
                new_pop.append(population[idx] )
            population = new_pop
        #!pickle.dump( population, open( "results/final_pop_{1}.p".format(max_fitness), "wb" ) )

if __name__ == '__main__':
    GeneticAlgorithm.run()
