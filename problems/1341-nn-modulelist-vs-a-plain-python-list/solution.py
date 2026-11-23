import torch
import torch.nn as nn


class LayerStack(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # TODO: build one nn.Linear(dims[i], dims[i+1]) per consecutive pair and
        #       store them in an nn.ModuleList assigned to self.layers
        self.layers = nn.ModuleList()
        
        for i in range(0, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        # TODO: apply each layer in order, with relu between consecutive layers
        #       but NOT after the final layer
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
            if idx != len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
        return x
