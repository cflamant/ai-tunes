"""
Author: Cedric Flamant
ai-tunes
An autoencoder tree backend for composing songs
"""
import numpy as np
import aitunes.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import tqdm


# TODO finish this parent module
# TODO Make inst_enc and inst_dec modules
class Autoencoder_Tree(nn.Module):

    def __init__(self, use_cuda=False, **kwargs):
        """Initialize this backend. Various hyperparameters are accepted, see
        the function set_hyperparams() for descriptions.

        Parameters
        ----------
        use_cuda : bool
            Whether to use CUDA or the CPU
        **kwargs :
            See set_hyperparams()

        Returns
        -------
        None
        """
        super(Autoencoder_Tree, self).__init__()

        self.p = self.set_hyperparams(**kwargs)
        self.use_cuda = use_cuda

        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("Backend: CUDA is not available.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.activation = self.p['activation']


    # TODO
    def set_hyperparams(self, hidden_dim=16,
                              in_dim=131,
                              out_dim=131,
                              embedding_dim=16,
                              num_layers=1,
                              max_len=128,
                              **kwargs):
        """Initialize dictionary of hyperparameters for model.

        Parameters
        ----------
        hidden_dim : int
            Size of the hidden state
        in_dim : int
            Size of the input dimension. Should be equal to the number of possible 
            characters in the string to be reversed, plus 3 for the starting symbol,
            separator, and terminal symbol.
        out_dim : int
            Size of the output dimension. Should be equal to the number of possible
            characters in the string to be reversed, plus 3 for the starting symbol,
            separator, and terminal symbol.
        embedding_dim : int
            Size of the embedding for the neural stack (dimension of value vectors)
        num_layers : int
            Number of LSTM layers to stack.
        max_len : int
            Maximum length of output. Can be made arbitrarily large if long strings
            are expected. Just used to avoid infinite output.

        Returns
        -------
        p : dictionary
            Dictionary containing hyperparameters by name

        """

        p = dict(hidden_dim=hidden_dim,
                      in_dim=in_dim,
                      out_dim=out_dim,
                      embedding_dim=embedding_dim,
                      num_layers=num_layers,
                      max_len=max_len)
        return p


class BasicInterpreter(nn.Module):
    """A simple interpreter layer that consists of a dense MLP.
    Each component of the output is between -1 and 1.
    """

    def __init__(self, use_cuda=False, **kwargs):
        """Initialize the interpreter layer. Various 
        hyperparameters are accepted, see the function
        set_hyperparams() for descriptions.

        Parameters
        ----------
        use_cuda : bool
            Whether to use CUDA or the CPU
        **kwargs :
            See set_hyperparams()

        Returns
        -------
        None
        """
        super(BasicInterpreter, self).__init__()

        self.p = self.set_hyperparams(**kwargs)
        self.use_cuda = use_cuda

        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("Backend: CUDA is not available.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.in_dim = self.p['in_dim']
        self.out_dim = self.p['out_dim']
        self.block_size = self.p['block_size']
        self.max_inst = self.p['max_inst']
        self.hidden_dim = self.p['hidden_dim']
        self.num_layers = self.p['num_layers']
        self.activation = self.p['activation']

        # List of modules to put in Sequential
        seq_list = []
        seq_list.append(nn.Linear(self.in_dim, self.hidden_dim))
        seq_list.append(self.activation)

        for i in range(1, self.num_layers):
            seq_list.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            seq_list.append(self.activation)

        seq_list.append(nn.Linear(self.hidden_dim, self.out_dim))
        seq_list.append(nn.Tanh())

        self.interpreter = nn.Sequential(*seq_list)


    def set_hyperparams(self, block_size=300,
                              max_inst=1,
                              out_dim=8,
                              hidden_dim=40,
                              num_layers=1,
                              activation=nn.Tanh(),
                              **kwargs):
        """Initialize dictionary of hyperparameters for the module.

        Parameters
        ----------
        block_size : int
            Number of samples (columns of a piano roll) that the
            interpreter block will receive.
        max_inst : int
            Maximum number of instruments (excluding drums)
        out_dim : int
            Size of the output dimension to the encoder.
        hidden_dim : int
            Size of the hidden dimension between linear layers.
        num_layers : int
            Number of hidden linear layers.
        activation : PyTorch activation module
            Activation to use after each linear layer. Output layer uses
            tanh.

        Returns
        -------
        p : dictionary
            Dictionary containing hyperparameters by name

        """
        if num_layers < 1:
            num_layers = 1

        in_dim = block_size * (24 * max_inst + 2)

        p = dict(in_dim=in_dim,
                 block_size=block_size,
                 max_inst=max_inst,
                 out_dim=out_dim,
                 hidden_dim=hidden_dim,
                 num_layers=num_layers,
                 activation=activation
                )
        return p


    def forward(self, X):
        """Forward propagation of the interpreter. 

        Parameters
        ----------
        X : tensor of dimension (minibatch, in_dim)
            Features consist of all chroma+octave vectors, and drum number,
            flattened across the number of samples per interpreter block.
            in_dim = block_size * (24 * max_inst + 2)

        Returns
        -------
        Xenc : tensor of dimension (minibatch, out_dim)
            Encoded representation of measure
        """
        Xenc = self.interpreter(X)
        return Xenc


class BasicExpresser(nn.Module):
    """A simple expresser layer that consists of a dense MLP.
    Output represents all chroma+octave vectors, and drum number, flattened
    across the number of samples per expresser block.
    """

    def __init__(self, use_cuda=False, **kwargs):
        """Initialize the expresser layer. Various 
        hyperparameters are accepted, see the function
        set_hyperparams() for descriptions.

        Parameters
        ----------
        use_cuda : bool
            Whether to use CUDA or the CPU
        **kwargs :
            See set_hyperparams()

        Returns
        -------
        None
        """
        super(BasicExpresser, self).__init__()

        self.p = self.set_hyperparams(**kwargs)
        self.use_cuda = use_cuda

        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("Backend: CUDA is not available.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.in_dim = self.p['in_dim']
        self.out_dim = self.p['out_dim']
        self.block_size = self.p['block_size']
        self.max_inst = self.p['max_inst']
        self.hidden_dim = self.p['hidden_dim']
        self.num_layers = self.p['num_layers']
        self.activation = self.p['activation']

        # List of modules to put in Sequential
        seq_list = []
        seq_list.append(nn.Linear(self.in_dim, self.hidden_dim))
        seq_list.append(self.activation)

        for i in range(1, self.num_layers):
            seq_list.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            seq_list.append(self.activation)

        seq_list.append(nn.Linear(self.hidden_dim, self.out_dim))
        # No final activation yet, the different parts of the output
        # vector have different activations (or no activation).

        self.expresser = nn.Sequential(*seq_list)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def set_hyperparams(self, in_dim=8,
                              block_size=300,
                              max_inst=1,
                              hidden_dim=40,
                              num_layers=1,
                              activation=nn.Tanh(),
                              **kwargs):
        """Initialize dictionary of hyperparameters for the module.

        Parameters
        ----------
        in_dim : int
            Size of the input dimension.
        block_size : int
            Number of samples (columns of a piano roll) that the
            interpreter block will receive.
        max_inst : int
            Maximum number of instruments (excluding drums)
        hidden_dim : int
            Size of the hidden dimension between linear layers.
        num_layers : int
            Number of hidden linear layers.
        activation : PyTorch activation module
            Activation to use after each linear layer. Output layer uses
            different activations for each part of the vector.

        Returns
        -------
        p : dictionary
            Dictionary containing hyperparameters by name

        """
        if num_layers < 1:
            num_layers = 1

        out_dim = block_size * (24 * max_inst + 2)

        p = dict(in_dim=in_dim,
                 block_size=block_size,
                 max_inst=max_inst,
                 out_dim=out_dim,
                 hidden_dim=hidden_dim,
                 num_layers=num_layers,
                 activation=activation
                )
        return p


    def forward(self, Xdec):
        """Forward propagation of the expresser. 

        Parameters
        ----------
        Xdec : tensor of dimension (minibatch, in_dim)
            The representation from the decoder that is to be expressed
            in terms of music.

        Returns
        -------
        Xhat : tensor of dimension (minibatch, out_dim)
            Output consists of all chroma+octave vectors, and drum number,
            flattened across the number of samples per expresser block.
            out_dim = block_size * (24 * max_inst + 2)
        """
        block_size = self.block_size
        Xpre = self.expresser(Xdec)
        Xhat = tensor.zeros_like(Xpre)

        # The first block_size * 12 elements are chroma vectors
        Xhat[:, :block_size*12] = self.sigmoid(Xpre[:, :block_size*12])
        # The second block_size * 12 elements are octave vectors
        # Restrict them to the range -5 to 5
        strt = block_size*12
        Xhat[:, strt:strt*2] = self.Tanh(Xpre[:, strt:strt*2]) * 5.
        # The next block_size elements are drum probabilities
        strt += block_size*12
        Xhat[:, strt:strt+block_size] = self.sigmoid(Xpre[:, strt:strt+block_size])
        # The final block_size elements are drum numbers
        strt += block_size
        Xhat[:, strt:] = self.tanh(Xpre[:, strt:]) * 30.
        return Xhat
