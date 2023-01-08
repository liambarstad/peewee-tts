import math
import torch
from torch import nn

class LSTMP(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, num_layers):
        super(LSTMP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.num_layers = num_layers
        
        self.first_cell = LSTMPCell(self.input_size, self.hidden_size, self.projection_size)
        self.hidden_cells = [ 
            LSTMPCell(self.hidden_size, self.hidden_size, self.projection_size)
            for n in range(num_layers)[1:]
        ]
        
    def forward(self, x):    
        x, hidden_state = self.first_cell(x)
        for hidden_cell in self.hidden_cells:
            x, hidden_state = hidden_cell(x, hidden_state)
        return x    
            
class LSTMPCell(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size):
        super(LSTMPCell, self).__init__()
        # optimized for speed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        
        self.W = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size * 4))
        
        self.init_weights()
        
    def forward(self, x, hidden_state=None):
        hx, cx = self._init_hidden_state(x) if not hidden_state else hidden_state
        
        
        return x, (hx, cx)
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def _init_hidden_state(self, x):
        return [
            torch.zeros(x.size()[0], self.hidden_size).to(x.device),
            torch.zeros(x.size()[0], self.hidden_size).to(x.device)
        ]    