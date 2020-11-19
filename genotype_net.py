
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

class MorphNet(nn.Module):
    def __init__(self, recurrent, input_dim=5, hidden_dim=64, number_state = 4):
        super(MorphNet, self).__init__()
        
        self.input_dim = input_dim

        self.recurrent = recurrent

        if (self.recurrent):
            self.linear1 = nn.Linear(input_dim, 32)
            self.lstm = nn.LSTM(32, hidden_dim) 
            self.linear2 = nn.Linear(hidden_dim, number_state) 

        else:
            self.linear1 = nn.Linear(input_dim, 64) 
            self.linear2 = nn.Linear(64, 64) 
            self.linear3 = nn.Linear(64, number_state)

    def forward(self,x,hidden=None):

        x = torch.FloatTensor(x)/3.0
        if (self.recurrent):
            x = torch.tanh( self.linear1(x) )
            x, hidden = self.lstm(x.unsqueeze(dim=0), hidden)
            x = self.linear2(x)

        else:
          x = F.tanh(self.linear1(x))
          x = F.tanh(self.linear2(x))
          x= self.linear3(x)
          #x = F.log_softmax(x)   
        return x, hidden

    def get_weights(self):
      return  parameters_to_vector(self.parameters() ).detach().numpy()

if __name__ == '__main__':
    main(sys.argv)