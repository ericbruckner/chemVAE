# chemVAE
A variational autoencoder (VAE) for molecular data. 

    Built using python3.8

## Software prerequisites:
- Python3 (Tested on version 3.8.5)
- numpy (Tested on version 1.19.2)
- pandas (Test on version 1.1.3)
- scikit-learn (Tested on version 0.23.2)
- Pytorch (Tested on version 1.8.0)
- RDKit (Test on version 2021.03.2)
- selfies (https://github.com/aspuru-guzik-group/selfies)


## Description
This package contains Python scripts to build and/or deploy a variational autoencoder (VAE) for chemical data. The encoder is based on a multilayer 1D convolutional network. The decoder is based on an LSTM RNN architecture. Typical steps in this process include the following:
- Convert a SMILES string to a SELFIES string
- One-hot encode a SELFIES string
- Train the VAE
- Load a VAE for the generation of new molecules

main.py is the main script that lets you do all the above as shown by the example (https://github.com/ericbruckner/chemVAE/blob/main/examples/example.ipynb)
