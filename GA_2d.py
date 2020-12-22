from __future__ import print_function

import multiprocessing
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle

from tools.read_write_voxelyze_2d import write_voxelyze_file, read_voxlyze_results

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
       #print(settings)
        sigma = settings['sigma']
        #print(sigma)

        np.random.seed(self.settings['seed'])

        filename="run_"+str(self.settings['seed'])
        if self.settings['expression']:
            filename+="_expression"
        
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
        #print(N)

        #N = max(10, N)
        elitism = 1 
        #print(self.weight_shape)
        
        for i in range(self.pop_size):
            #print(i)
            w = torch.from_numpy( np.random.normal(0, 1, self.weight_shape)*0.1 ).float()
            population.append( w )
            #print(w)
        #print(population[0])
        #print(len(population))
        #print(w)
        #print(len(population),type(population))
        max_fitness = 0

        cmap = colors.ListedColormap(['w','c','lime','b','r'])
            
        
        pool = multiprocessing.Pool(self.num_threads)

       
        for epoch in range(0, generations):
            print("Epoch ",epoch, "Generations", generations, "Population size", len(population))
            #print((population, self.settings) )
            #print(self.settings)
            
            #fitness, sim, env, generations, individual_id, morphogens 
            results = pool.map(self.fitness_function, [ (it, epoch, self.settings) for it in enumerate(population) ])
            #print(results, type(results))
            fitness = [row[0] for row in results]
            sim = [row[1] for row in results]
            env = [row[2] for row in results]
            individual_id = [row[4] for row in results]
            morphogens = [row[5] for row in results]
            hidden_states_batched_A = [row[6] for row in results]
            hidden_states_batched_B = [row[7] for row in results]
            dev_states = [row[8] for row in results]
            #print(len(dev_states))
        
            fitness_all.append(fitness)
            #print(fitness_all)
            #print(type([ (i, self.settings) for i in population ]),len([ (i, self.settings) for i in population ]))
            #print(len(fitness))
            #print(type(population))
            #print(fitness)
            #print([-f for f in fitness])
            #Get individuales id
            sort_idx = np.argsort([-f for f in fitness])
            #print(sort_idx)
            #print(results)
            max_gen_f = np.max(fitness)
            #print(np.max(fitness))
    
            if (max_gen_f>max_fitness):
                max_fitness = max_gen_f
                #print(sort_idx[0])
                write_voxelyze_file(sim[sort_idx[0]], env[sort_idx[0]], epoch, individual_id[sort_idx[0]], morphogens[sort_idx[0]], max_fitness, self.settings['im_size'], self.settings['run_directory'], self.settings['run_name'])
                torch.save(population[sort_idx[0]], "{0}/weightFiles/GA_saved_weights_gen_{1}_{2}.dat".format(self.settings['run_directory'],epoch, max_fitness))
                dev_states = np.asarray(dev_states)
                np.save('{0}/dev_stateFiles/dev_states_gen_{1}_id_{2}'.format(self.settings['run_directory'], epoch, sort_idx[0]), dev_states[sort_idx[0]])
                #filename = "{0}/GA_saved_weights_gen_{1}_{2}".format(self.folder,epoch, max_fitness)
                #m = self.fitness_function( (population[sort_idx[0] ], self.settings), True, filename)
                #print(dev_states.shape, dev_states[0].shape, dev_states[0][0].shape,len(dev_states), len(dev_states[1]))
                dev_states = dev_states.reshape(len(population),len(dev_states[1]),self.settings['im_size'],self.settings['im_size'])
                #print(dev_states.shape, dev_states[0].shape, dev_states[0][0].shape,len(dev_states), len(dev_states[1]))
            
                for i in range(1,len(dev_states[1])+1):
                    #print(i)
                    plt.subplot(2, len(dev_states[1])/2, i)
                    plt.axis('off')
                    imgplot = plt.imshow(dev_states[sort_idx[0]][i-1], cmap = cmap, interpolation='none') 
                plt.savefig('{0}/dev_stateFiles/gen{1}_id{2}.eps'.format(self.settings['run_directory'],epoch, sort_idx[0]))
                plt.close()
            
                if LOG:
                    self.writer.add_image("Image", m.transpose(2, 0, 1), epoch)
            #print(sort_idx)

            print ("Max fitness ", max_gen_f," mean ",np.mean(fitness), " best ever ",max_fitness)#, " ", len(population) )
            #Save to file
        
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
                plt.savefig('{0}/epoch{1}.eps'.format(self.settings['run_directory'],epoch))

                torch.save(population, "{0}/model_stateFiles/weights_gen_{1}_{2}.dat".format(self.settings['run_directory'],epoch, max_fitness))
                np.save('{0}/model_stateFiles/morphogens_gen_{1}'.format(self.settings['run_directory'],epoch),morphogens)
                torch.save(hidden_states_batched_A,'{0}/model_stateFiles/hidden_states_batched_A_gen_{1}.pt'.format(self.settings['run_directory'],epoch))
                torch.save(hidden_states_batched_B,'{0}/model_stateFiles/hidden_states_batched_B_gen_{1}.pt'.format(self.settings['run_directory'],epoch))

                str_ = str(fitness_log)
                str_1 = str(fitness_all)
                with open("{0}/fitnessFiles/fitness_log_gen_{1}.txt".format(self.settings['run_directory'], epoch), 'wt') as f:
                    f.write(str_)
                with open("{0}/fitnessFiles/fitness_all_gen_{1}.txt".format(self.settings['run_directory'], epoch), 'wt') as g:
                    g.write(str_1)
               

            new_pop = []
            for idx in range(self.pop_size-elitism):

                #Select indivdual from the top N
                i = np.random.randint(0, N )
                p = population[ sort_idx[i]]
                #print(i, p)

                new_ind = p + torch.from_numpy( np.random.normal(0, 1, self.weight_shape) * self.sigma).float()

                new_pop.append(new_ind)
            #print(len(new_pop))

            for idx in sort_idx[:elitism]:
                new_pop.append(population[idx] )
            #print(sort_idx, sort_idx[:elitism])
            population = new_pop


            
                #print(type(morphogens[sort_idx[0]]),)
                #print(type(hidden_states_batched_A[sort_idx[0]]))
                #print(type(hidden_states_batched_B[sort_idx[0]]))

        #!pickle.dump( population, open( "results/final_pop_{1}.p".format(max_fitness), "wb" ) )

if __name__ == '__main__':
    GeneticAlgorithm.run()
