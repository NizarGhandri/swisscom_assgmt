import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, from_numpy
from torch.utils.data import Dataset

SCALED_COLS = ["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365", "longitude", "latitude"]
BINARIZED_COLS = ["neighbourhood", "room_type"]
TARGET = ["log_price", "neighbourhood_group"]
FEATURES = SCALED_COLS




class SimpleNeuralNetwork(nn.Module):

    def __init__(self, *args, separate = False, **kwargs):
        super(SimpleNeuralNetwork, self).__init__()
        self.separate = separate
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(args[:-1], args[1:])])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(out) for out in args[1:-1]])


    def forward(self, x):
        if self.separate:
            x_clf, x_re = x
            x_ = []
            for inp in [x_clf, x_re]:
                x__ = self.layer[0](x__.float())
                x__ = self.batch_norm[0](x__)
                x__ = nn.ReLU()(x__)
                x__ = nn.Dropout()(x__)
            
            x_ = torch.cat(x_, dim=0)
                
            for layer, batch_norm in zip(self.layers[1:], self.batch_norms[1:]): 
                x_ = layer(x_)
                x_ = batch_norm(x_)
                x_ = nn.ReLU()(x_)
                x_ = nn.Dropout()(x_)
        
            
        else:
            x_ = x.float()
            
            for layer, batch_norm in zip(self.layers, self.batch_norms): 
                x_ = layer(x_)
                x_ = batch_norm(x_)
                x_ = nn.ReLU()(x_)
                x_ = nn.Dropout()(x_)
        
         
            
            
        x_ = self.layers[-1](x_)
        x_ = torch.cat([nn.ReLU()(x_[:, :1]), x_[:, 1:]], 1) 
        return x_
    

class SeparateNeuralNetwork(nn.Module):

    def __init__(self, reg_size, clf_size, *args, **kwargs):
        super(SimpleNeuralNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(args[:-1], args[1:])])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(out) for out in args[1:-1]])
        self.layer_1_reg = nn.Linear(reg_size, out)


    def forward(self, x):
        x_clf, x_re = x
        x_ = []
        for inp in [x_clf, x_re]:
            x__ = self.layer[0](x__.float())
            x__ = self.batch_norm[0](x__)
            x__ = nn.ReLU()(x__)
            x__ = nn.Dropout()(x__)
        
        x_ = torch.cat(x_, dim=0)
            
        for layer, batch_norm in zip(self.layers[1:], self.batch_norms[1:]): 
            x_ = layer(x_)
            x_ = batch_norm(x_)
            x_ = nn.ReLU()(x_)
            x_ = nn.Dropout()(x_)
        
            
        
         
            
            
        x_ = self.layers[-1](x_)
        x_ = torch.cat([nn.ReLU()(x_[:, :1]), x_[:, 1:]], 1) 
        return x_


class SeperableDataset(Dataset):
    
    
    def __init__(self, X, y, seperable):
        
        self.X = X
        self.y = y
        self.seperable = seperable
        self.clf_ = ["longitude", "latitude"] + list(filter(lambda x: x.startswith("neighbourhood"), list(X.columns)))
        self.reg_ = [f for f in X.columns if not f in ["longitude", "latitude"]]
        
        
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, x):
        if self.seperable:
            return from_numpy(self.X.iloc[x].values), from_numpy(self.y.iloc[x].values)
        else:
            return from_numpy(self.X[self.clf_].iloc[x].values), from_numpy(self.X[self.reg_].iloc[x].values), from_numpy(y.iloc[0].values)