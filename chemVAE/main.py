# import libraries
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from datetime import date
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, optim

from rdkit import Chem

import selfies as sf

# main functions for data processing
def smiles2selfies(smiles):
    selfies = []
    for smi in smiles:
        if '.' not in smi:
            encoded_selfies = sf.encoder(smi)
            if encoded_selfies is not None:
                selfies.append(encoded_selfies)
                
    return selfies

def onehotSELFIES(selfies):
    # get selfies alphabet
    alphabet = sf.get_alphabet_from_selfies(selfies)
    alphabet.add('[nop]')  # '[nop]' is a special padding symbol
    alphabet = list(sorted(alphabet))

    pad_to_len = max(sf.len_selfies(s) for s in selfies)  
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    idx_to_symbol = {ch: ii for ii, ch in symbol_to_idx.items()}
    
    # embed list of characters to list of integers
    embed_selfies = []
    for s in selfies:
        embed = sf.selfies_to_encoding(s,
                                 vocab_stoi=symbol_to_idx,
                                 pad_to_len=pad_to_len,
                                 enc_type='label')    
        embed_selfies.append(embed)

        # one hot encode
    dict_size = len(symbol_to_idx)
    seq_len = pad_to_len
    data_size = len(embed_selfies)
    sequence = embed_selfies

    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((data_size, dict_size,seq_len), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(data_size):
        for u in range(seq_len):
            features[i, sequence[i][u], u] = 1

    onehot_selfies = features
    
    return onehot_selfies, idx_to_symbol

# VAE model compsed of 1DConv encoder and LSTM decoder
class VAE(nn.Module):

    def __init__(self, params):

        super(VAE, self).__init__()
        
        # Load Model Parameters
        self.num_characters = params['num_characters']
        self.max_seq_len = params['seq_length']
        self.in_dimension = params['num_characters']*params['seq_length']
        self.output_dimension = params['seq_length']
        
        self.num_conv_layers = params['num_conv_layers']
        self.layer1_filters = params['layer1_filters']
        self.layer2_filters = params['layer2_filters']
        self.layer3_filters = params['layer3_filters']
        self.layer4_filters = params['layer4_filters']
        self.kernel1_size = params['kernel1_size']
        self.kernel2_size = params['kernel2_size']
        self.kernel3_size = params['kernel3_size']
        self.kernel4_size = params['kernel4_size']
        self.lstm_stack_size = params['lstm_stack_size']
        self.lstm_num_neurons = params['lstm_num_neurons']
        self.latent_dimensions = params['latent_dimensions']
        self.batch_size = params['batch_size']
        
        # Conv1D encoding layers
        self.convl1 = nn.Conv1d(self.num_characters, self.layer1_filters, self.kernel1_size, padding = self.kernel1_size//2)
        self.convl2 = nn.Conv1d(self.layer1_filters, self.layer2_filters, self.kernel2_size, padding = self.kernel2_size//2)
        self.convl3 = nn.Conv1d(self.layer2_filters, self.layer3_filters, self.kernel3_size, padding = self.kernel3_size//2)
        self.convl4 = nn.Conv1d(self.layer3_filters, self.layer4_filters, self.kernel4_size, padding = self.kernel4_size//2)



        # Linear layers to connect convolutional layers to mu and logvar
        if self.num_conv_layers == 1:
            self.fc_mu = nn.Linear(self.layer1_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer1_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
        elif self.num_conv_layers == 2:
            self.fc_mu = nn.Linear(self.layer2_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer2_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
        elif self.num_conv_layers == 3:
            self.fc_mu = nn.Linear(self.layer3_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer3_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
        elif self.num_conv_layers == 4:
            self.fc_mu = nn.Linear(self.layer4_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer4_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
            

        # LSTM decoding layers
        self.decode_RNN = nn.LSTM(
            input_size = self.latent_dimensions,
            hidden_size = self.lstm_num_neurons,
            num_layers = self.lstm_stack_size,
            batch_first = True,
            bidirectional = True)

        self.decode_FC = nn.Sequential(
            nn.Linear(2*self.lstm_num_neurons, self.output_dimension),
        )
        
        self.prob = nn.LogSoftmax(dim=1)
        
    @staticmethod
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden =  weight.new_zeros(self.lstm_stack_size, batch_size, self.lstm_num_neurons).zero().to(device)
        return hidden                                
                                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std
    
    def encoder(self, x):
        if self.num_conv_layers == 1:
            x = F.relu(self.convl1(x))
        elif self.num_conv_layers == 2:
            x = F.relu(self.convl1(x))
            x = F.relu(self.convl2(x))
        elif self.num_conv_layers == 3:
            x = F.relu(self.convl1(x))
            x = F.relu(self.convl2(x))
            x = F.relu(self.convl3(x))
        elif self.num_conv_layers == 4:
            x = F.relu(self.convl1(x))
            x = F.relu(self.convl2(x))
            x = F.relu(self.convl3(x))
            x = F.relu(self.convl4(x))

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
    
    def decoder(self, z):
        rz = z.unsqueeze(1).repeat(1, self.num_characters, 1)
        l1, h = self.decode_RNN(rz)
        decoded = self.decode_FC(l1)
        x_hat = decoded
        
        return x_hat

    def forward(self, x):
        # Get results of encoder network
        x = x.squeeze(dim=1)
        z, mu, logvar = self.encoder(x)

        # Get results of decoder network
        x_hat = self.decoder(z)

                 
        return x_hat, z, mu, logvar
    
# VAE loss function #
def loss_function(recon_x, x, mu, logvar, KLD_alpha):   
    #BCE = F.binary_cross_entropy(recon_x, x.squeeze(dim=1), reduction='sum')
    
    inp = recon_x
    target = torch.argmax(x, dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    BCE = criterion(inp, target)
    KLD = -0.5 * torch.mean(1. + logvar - mu.pow(2) - logvar.exp())
    
    #return BCE + KLD_alpha*KLD
    return BCE, KLD_alpha, KLD


# VAE training loop
def train(model, train_loader, optimizer, device, epoch, KLD_alpha):
    LOG_INTERVAL = 100
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)            
        recon_data, z, mu, logvar = model(data)
        BCE, KLD_alpha, KLD = loss_function(recon_data, data.squeeze(dim=1), mu, logvar, KLD_alpha)
        loss = BCE + KLD_alpha*KLD
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.5f}'.format(epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss / len(train_loader.dataset), BCE, KLD_alpha, KLD

# VAE testing loop
def test(model, test_loader, optimizer, device, epoch, KLD_alpha):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_data, z, mu, logvar = model(data)
            BCE, KLD_alpha, KLD = loss_function(recon_data, data.squeeze(dim=1), mu, logvar, KLD_alpha)
            cur_loss = BCE + KLD_alpha*KLD
            test_loss += cur_loss

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.5f}'.format(test_loss))

    return test_loss

# custom dataset
class SELFIES_Dataset(Dataset):
    def __init__(self, input_seq, target_seq, transform=None):
        self.X = input_seq
        self.y = target_seq
        self.transforms = transform
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, idx):
        
        if self.transforms:
            X = self.transforms(self.X[idx])
            y = self.transforms(self.y[idx])
            return (X, y)

        else:
            return (self.X[idx], self.y[idx])