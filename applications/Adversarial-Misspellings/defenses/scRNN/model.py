import torch
from torch import nn
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class ScRNN(nn.Module):
    def __init__(self, char_vocab_size, hdim, output_dim):
        super(ScRNN, self).__init__()
        """ layers """
        self.lstm = nn.LSTM(3*char_vocab_size, hdim, 1, batch_first=True,
            bidirectional=True) 
        self.linear = nn.Linear(2*hdim, output_dim)



    """ size(inp) --> BATCH_SIZE x MAX_SEQ_LEN x EMB_DIM 
    """
    def forward(self, inp, lens):
        packed_input = pack_padded_sequence(inp, lens, batch_first=True)
        packed_output, _ = self.lstm(packed_input)
        h, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.linear(h) # out is batch_size x max_seq_len x class_size
        out = out.transpose(dim0=1, dim1=2)
        return out # out is batch_size  x class_size x  max_seq_len



class ElmoScRNN(nn.Module):
    def __init__(self, char_vocab_size, hdim, output_dim):
        super(ElmoScRNN, self).__init__()
        self.elmo_hdim = 1024
        self.weight_file = "../elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.options_file = "../elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        """ layers """
        self.elmo = Elmo(self.options_file, self.weight_file, 1)
        self.lstm = nn.LSTM(3*char_vocab_size, hdim, 1, batch_first=True,
            bidirectional=True)
        self.linear = nn.Linear(2*hdim + self.elmo_hdim, output_dim)



    #TODO: go away from the assumption that the batch size is 1. 
    """ size(inp) --> BATCH_SIZE x MAX_SEQ_LEN x EMB_DIM
    """
    def forward(self, inp, elmo_inp, lens):
        packed_input = pack_padded_sequence(inp, lens, batch_first=True)
        packed_output, _ = self.lstm(packed_input)
        h, _ = pad_packed_sequence(packed_output, batch_first=True) # h is BATCH_SIZE x MAX_SEQ_LEN x hdim
        h_e = self.elmo(elmo_inp)['elmo_representations'][0] # h_e is BATCH_SIZE X MAX_SEQ_LEN x 1024

        h = torch.cat((h, h_e), 2) # concat along the last dim

        out = self.linear(h) # out is batch_size x max_seq_len x class_size
        out = out.transpose(dim0=1, dim1=2)
        return out # out is batch_size  x class_size x  max_seq_len



"""
This is a vanilla model, which takes the ELMO representations for each token,
     and tries to reconstruct each word using the (oft manipulated) word
"""
class ElmoRNN(nn.Module):
    def __init__(self, output_dim):
        super(ElmoRNN, self).__init__()
        self.elmo_hdim = 1024
        self.weight_file = "../elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.options_file = "../elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        """ layers """
        self.elmo = Elmo(self.options_file, self.weight_file, 1)
        self.linear = nn.Linear(self.elmo_hdim, output_dim)


    """ size(inp) --> BATCH_SIZE x MAX_SEQ_LEN x EMB_DIM
    """
    def forward(self, inp):
        h = self.elmo(inp)['elmo_representations'][0] # h_e is BATCH_SIZE X MAX_SEQ_LEN x 1024

        out = self.linear(h) # out is batch_size x max_seq_len x class_size
        out = out.transpose(dim0=1, dim1=2) # flip the second and the third dimensions
        return out # out is batch_size  x class_size x  max_seq_len
