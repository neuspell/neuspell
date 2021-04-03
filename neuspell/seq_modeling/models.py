import os

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

from .util import is_module_available, get_module_or_attr
from ..commons import ALLENNLP_ELMO_PRETRAINED_FOLDER


def get_pretrained_elmo(elmo_options_file=None, elmo_weights_file=None):
    if not is_module_available("allennlp"):
        raise ImportError(
            "install `allennlp` by running `pip install -r extras-requirements.txt`. See `README.md` for more info.")

    Elmo = get_module_or_attr("allennlp.modules.elmo", "Elmo")

    local_options_file, local_weights_file = None, None
    if os.path.exists(ALLENNLP_ELMO_PRETRAINED_FOLDER):
        local_options_file = os.path.join(ALLENNLP_ELMO_PRETRAINED_FOLDER,
                                          "elmo_2x4096_512_2048cnn_2xhighway_options.json")
        local_weights_file = os.path.join(ALLENNLP_ELMO_PRETRAINED_FOLDER,
                                          "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")

    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weights_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo_options_file = elmo_options_file or local_options_file or options_file  # or os.environ.get('ELMO_OPTIONS_FILE_PATH', None)
    elmo_weights_file = elmo_weights_file or local_weights_file or weights_file  # or os.environ.get('ELMO_WEIGHTS_FILE_PATH', None)
    # neuspell.seq_modeling.models.get_pretrained_elmo()
    return Elmo(elmo_options_file, elmo_weights_file, 1)  # 1 for setting device="cuda:0" else 0


def get_pretrained_bert(pretrained_name_or_path="bert-base-cased"):
    return transformers.BertModel.from_pretrained(pretrained_name_or_path)


#################################################
# CharCNNWordLSTMModel(CharCNNModel)
#################################################

class CharCNNModel(nn.Module):
    def __init__(self, nembs, embdim, padding_idx, filterlens, nfilters):
        super(CharCNNModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim, padding_idx=padding_idx)
        # torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # Unsqueeze [BS, MAXSEQ, EMDDIM] as [BS, 1, MAXSEQ, EMDDIM] and send as input
        self.convmodule = nn.ModuleList()
        for length, n in zip(filterlens, nfilters):
            self.convmodule.append(
                nn.Sequential(
                    nn.Conv2d(1, n, (length, embdim), padding=(length - 1, 0), dilation=1, bias=True,
                              padding_mode='zeros'),
                    nn.ReLU()
                )
            )
        # each conv outputs [BS, nfilters, MAXSEQ, 1]

    def forward(self, batch_tensor):
        batch_size = len(batch_tensor)

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)

        # [BS, max_seq_len, emb_dim]->[BS, 1, max_seq_len, emb_dim]
        embs_unsqueezed = torch.unsqueeze(embs, dim=1)

        # [BS, 1, max_seq_len, emb_dim]->[BS, out_channels, max_seq_len, 1]->[BS, out_channels, max_seq_len]
        conv_outputs = [conv(embs_unsqueezed).squeeze(3) for conv in self.convmodule]

        # [BS, out_channels, max_seq_len]->[BS, out_channels]
        maxpool_conv_outputs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]

        # cat( [BS, out_channels] )->[BS, sum(nfilters)]
        source_encodings = torch.cat(maxpool_conv_outputs, dim=1)
        return source_encodings


class CharCNNWordLSTMModel(nn.Module):
    def __init__(self, nchars, char_emb_dim, char_padding_idx, padding_idx, output_dim):
        super(CharCNNWordLSTMModel, self).__init__()

        # cnn module
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,sum(nfilters)]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, sum(nfilters)]
        nfilters, filtersizes = [50, 100, 100, 100], [2, 3, 4, 5]
        self.cnnmodule = CharCNNModel(nchars, char_emb_dim, char_padding_idx, filtersizes, nfilters)
        self.cnnmodule_outdim = sum(nfilters)

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 512, 2
        self.lstmmodule = nn.LSTM(self.cnnmodule_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.lstmmodule_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    def forward(self,
                batch_idxs: "list[pad_sequence]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1):

        batch_size = len(batch_idxs)

        # cnn
        cnn_encodings = [self.cnnmodule(pad_sequence_) for pad_sequence_ in batch_idxs]
        cnn_encodings = pad_sequence(cnn_encodings, batch_first=True, padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = cnn_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                      batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss


#################################################
# CharLSTMWordLSTMModel(CharLSTMModel)
#################################################

class CharLSTMModel(nn.Module):
    def __init__(self, nembs, embdim, padding_idx, hidden_size, num_layers, bidirectional, output_combination):
        super(CharLSTMModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim, padding_idx=padding_idx)
        # torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # lstm module
        # expected input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        self.lstmmodule = nn.LSTM(embdim, hidden_size, num_layers, batch_first=True, dropout=0.3,
                                  bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output
        assert output_combination in ["end", "max", "mean"], print(
            'invalid output_combination; required one of {"end","max","mean"}')
        self.output_combination = output_combination

    def forward(self, batch_tensor, batch_lengths):

        batch_size = len(batch_tensor)
        # print("************ stage 2")

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        embs_packed = pack_padded_sequence(embs, batch_lengths, batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(embs_packed)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # [BS, max_seq_len, self.lstmmodule_outdim]->[BS, self.lstmmodule_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstmmodule_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError

        return source_encodings


class CharLSTMWordLSTMModel(nn.Module):
    def __init__(self, nchars, char_emb_dim, char_padding_idx, padding_idx, output_dim):
        super(CharLSTMWordLSTMModel, self).__init__()

        # charlstm module
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,charlstm_outputdim]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, charlstm_outputdim]
        hidden_size, num_layers, bidirectional, output_combination = 256, 1, True, "end"
        self.charlstmmodule = CharLSTMModel(nchars, char_emb_dim, char_padding_idx, hidden_size, num_layers,
                                            bidirectional, output_combination)
        self.charlstmmodule_outdim = self.charlstmmodule.lstmmodule_outdim

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 512, 2
        self.lstmmodule = nn.LSTM(self.charlstmmodule_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.lstmmodule_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    def forward(self,
                batch_idxs: "list[pad_sequence]",
                batch_char_lengths: "list[tensor]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1):

        batch_size = len(batch_idxs)
        # print("************ stage 1")

        # charlstm
        charlstm_encodings = [self.charlstmmodule(pad_sequence_, lens) for pad_sequence_, lens in
                              zip(batch_idxs, batch_char_lengths)]
        charlstm_encodings = pad_sequence(charlstm_encodings, batch_first=True, padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = charlstm_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                      batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss


#################################################
# SCLSTM
#################################################

class SCLSTM(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim):
        super(SCLSTM, self).__init__()
        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 512, 2
        self.lstmmodule = nn.LSTM(screp_dim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.4, bidirectional=bidirectional)  # 0.3 or 0.4
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.5)  # 0.4 or 0.5
        self.dense = nn.Linear(self.lstmmodule_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    def forward(self,
                batch_screps: "list[pad_sequence]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1):

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = batch_screps
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                      batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss


#################################################
# ElmoSCLSTM
#################################################

class ElmoSCLSTM(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim, early_concat=True):
        super(ElmoSCLSTM, self).__init__()

        self.early_concat = early_concat  # if True, (elmo+sc)->lstm->linear, else ((sc->lstm)+elmo)->linear

        self.elmo = get_pretrained_elmo()
        self.elmomodule_outdim = 1024

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 512, 2
        if self.early_concat:
            self.lstmmodule_indim = screp_dim + self.elmomodule_outdim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim
        else:
            self.lstmmodule_indim = screp_dim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim + self.elmomodule_outdim

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.encodings_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    def forward(self,
                batch_screps: "list[pad_sequence]",
                batch_lengths: "tensor",
                elmo_inp: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1,
                beam_search=False):

        if aux_word_embs is not None:
            raise Exception("dimensions of aux_word_embs not used in __init__()")

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0)

        # elmo
        elmo_encodings = self.elmo(elmo_inp)['elmo_representations'][0]  # BS X max_nwords x 1024

        if self.early_concat:

            # concat aux_embs
            # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
            intermediate_encodings = torch.cat((batch_screps, elmo_encodings), dim=2)
            if aux_word_embs is not None:
                intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

            # lstm
            # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
            intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                          batch_first=True, enforce_sorted=False)
            lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
            lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

            # out
            final_encodings = lstm_encodings

        else:

            # concat aux_embs
            # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
            intermediate_encodings = batch_screps
            if aux_word_embs is not None:
                intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

                # lstm
            # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
            intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                          batch_first=True, enforce_sorted=False)
            lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
            lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

            # out
            final_encodings = torch.cat((lstm_encodings, elmo_encodings), dim=2)

        # dense
        # [BS,max_nwords,self.encodings_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(final_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]

            if not beam_search:
                if topk > 1:
                    topk_probs, topk_inds = \
                        torch.topk(probs, topk, dim=-1, largest=True,
                                   sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
                elif topk == 1:
                    topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]
                else:
                    raise Exception("topk can be one of a value>=1")

                # Note that for those positions with padded_idx,
                #   the arg_max_prob above computes a index because 
                #   the bias term leads to non-uniform values in those positions

                return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()

            else:
                topk_probs, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
                return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy(), topk_probs.cpu().detach().numpy()

        return loss


#################################################
# SubwordElmo
#################################################

class SubwordElmo(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim):
        super(SubwordElmo, self).__init__()

        self.elmo = get_pretrained_elmo()
        self.elmomodule_outdim = 1024

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.elmomodule_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    def forward(self,
                batch_elmo_inp: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1,
                beam_search=False):

        # cnn
        batch_size = len(batch_elmo_inp)

        # elmo
        elmo_encodings = self.elmo(batch_elmo_inp)['elmo_representations'][0]  # BS X max_nwords x 1024

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = elmo_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(intermediate_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]

            if not beam_search:
                if topk > 1:
                    topk_probs, topk_inds = \
                        torch.topk(probs, topk, dim=-1, largest=True,
                                   sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
                elif topk == 1:
                    topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]
                else:
                    raise Exception("topk can be one of a value>=1")

                # Note that for those positions with padded_idx,
                #   the arg_max_prob above computes a index because 
                #   the bias term leads to non-uniform values in those positions

                return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()

            else:
                topk_probs, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
                return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy(), topk_probs.cpu().detach().numpy()

        return loss


#################################################
# ElmoSCTransformer
#################################################

'''
References:
- See https://pytorch.org/docs/stable/nn.html#transformerencoder for multiple encoder layers
    - See the arg src_key_padding_mask in .forward()
    - See https://github.com/pytorch/pytorch/blob/a5b509985a37127fb52fbfdee85c7b336cd8d2c1/torch/nn/modules/activation.py#L799
- See https://pytorch.org/docs/stable/tensors.html for torch.unit8
- See http://jalammar.github.io/illustrated-transformer/ for the architecture understanding
- See https://discuss.pytorch.org/t/embed-dim-must-be-divisible-by-num-heads/54394/2
    - AssertionError: embed_dim must be divisible by num_heads

Modifications compared to ElmoSCLSTM:
- bidirectional=True is replaced with nhead=2
- hidden_size is used as dim_feedforward in transformer encoder
- arg src_key_padding_mask added to .forward() in ElmoSCTransformer
    - thus, sctrans_tokenize() was created in helpers.py
'''


class ElmoSCTransformer(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim):
        super(ElmoSCTransformer, self).__init__()

        self.elmo = get_pretrained_elmo()

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        nhead, hidden_size, nlayers = 8, 1024, 1
        self.transmodule_indim = screp_dim + 2 + 1024
        print(self.transmodule_indim)
        # self.transmodule = nn.LSTM(self.transmodule_indim, hidden_size, nlayers, 
        #                             batch_first=True, dropout=0.3, bidirectional=bidirectional)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.transmodule_indim, nhead=nhead,
                                                         dim_feedforward=hidden_size, dropout=0.3, activation='relu')
        self.transmodule = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.transmodule_outdim = self.transmodule_indim

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.transmodule_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    def forward(self,
                batch_screps: "list[pad_sequence]",
                src_key_padding_mask: "tensor",
                batch_lengths: "tensor",
                elmo_inp: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1):

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0)

        # elmo
        elmo_encodings = self.elmo(elmo_inp)['elmo_representations'][0]  # BS X max_nwords x 1024

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = torch.cat((batch_screps, elmo_encodings), dim=2)
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # transformer
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.transmodule_outdim]
        intermediate_encodings = intermediate_encodings.permute(1, 0, 2)
        transformer_encodings = self.transmodule(intermediate_encodings, src_key_padding_mask=src_key_padding_mask)
        transformer_encodings = transformer_encodings.permute(1, 0, 2)

        # dense
        # [BS,max_nwords,self.transmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(transformer_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss


#################################################
# BertSCLSTM
#################################################

"""
import transformers
import torch
"""


class BertSCLSTM(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim, early_concat=True):
        super(BertSCLSTM, self).__init__()

        self.bert_dropout = torch.nn.Dropout(0.2)
        self.bert_model = get_pretrained_bert()
        self.bertmodule_outdim = self.bert_model.config.hidden_size
        self.early_concat = early_concat  # if True, (bert+sc)->lstm->linear, else ((sc->lstm)+bert)->linear
        # Uncomment to freeze BERT layers
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 512, 2
        if self.early_concat:
            self.lstmmodule_indim = screp_dim + self.bertmodule_outdim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim
        else:
            self.lstmmodule_indim = screp_dim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim + self.bertmodule_outdim

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.encodings_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_merged_encodings(self, bert_seq_encodings, seq_splits, mode='avg'):
        bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
        bert_seq_encodings = bert_seq_encodings[1:-1, :]
        # a tuple of tensors
        split_encoding = torch.split(bert_seq_encodings, seq_splits, dim=0)
        batched_encodings = pad_sequence(split_encoding, batch_first=True, padding_value=0)
        if mode == 'avg':
            seq_splits = torch.tensor(seq_splits).reshape(-1, 1).to(self.device)
            out = torch.div(torch.sum(batched_encodings, dim=1), seq_splits)
        elif mode == "add":
            out = torch.sum(batched_encodings, dim=1)
        else:
            raise Exception("Not Implemented")
        return out

    def forward(self,
                batch_screps: "list[pad_sequence]",
                batch_lengths: "tensor",
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: "list[list[int]]",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1):

        if aux_word_embs is not None:
            raise Exception("dimensions of aux_word_embs not used in __init__()")

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0)

        # bert
        # BS X max_nsubwords x self.bertmodule_outdim
        bert_encodings, cls_encoding = self.bert_model(**batch_bert_dict, return_dict=False)
        bert_encodings = self.bert_dropout(bert_encodings)
        # BS X max_nwords x self.bertmodule_outdim
        bert_merged_encodings = pad_sequence(
            [self.get_merged_encodings(bert_seq_encodings, seq_splits, mode='avg') \
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)],
            batch_first=True,
            padding_value=0
        )

        if self.early_concat:

            # concat aux_embs
            # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
            intermediate_encodings = torch.cat((batch_screps, bert_merged_encodings), dim=2)
            if aux_word_embs is not None:
                intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

            # lstm
            # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
            intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                          batch_first=True, enforce_sorted=False)
            lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
            lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

            # out
            final_encodings = lstm_encodings

        else:

            # concat aux_embs
            # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
            intermediate_encodings = batch_screps
            if aux_word_embs is not None:
                intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

                # lstm
            # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
            intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                          batch_first=True, enforce_sorted=False)
            lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
            lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

            # out
            final_encodings = torch.cat((lstm_encodings, bert_merged_encodings), dim=2)

        # dense
        # [BS,max_nwords,self.encodings_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(final_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss


#################################################
# SubwordBert
#################################################

class SubwordBert(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim):
        super(SubwordBert, self).__init__()

        self.bert_dropout = torch.nn.Dropout(0.2)
        self.bert_model = get_pretrained_bert()
        self.bertmodule_outdim = self.bert_model.config.hidden_size
        # Uncomment to freeze BERT layers
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False

        # output module
        assert output_dim > 0
        # self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.bertmodule_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_merged_encodings(self, bert_seq_encodings, seq_splits, mode='avg'):
        bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
        bert_seq_encodings = bert_seq_encodings[1:-1, :]
        # a tuple of tensors
        split_encoding = torch.split(bert_seq_encodings, seq_splits, dim=0)
        batched_encodings = pad_sequence(split_encoding, batch_first=True, padding_value=0)
        if mode == 'avg':
            seq_splits = torch.tensor(seq_splits).reshape(-1, 1).to(self.device)
            out = torch.div(torch.sum(batched_encodings, dim=1), seq_splits)
        elif mode == "add":
            out = torch.sum(batched_encodings, dim=1)
        else:
            raise Exception("Not Implemented")
        return out

    def forward(self,
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: "list[list[int]]",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1):

        # cnn
        batch_size = len(batch_splits)

        # bert
        # BS X max_nsubwords x self.bertmodule_outdim
        bert_encodings, cls_encoding = self.bert_model(**batch_bert_dict, return_dict=False)
        bert_encodings = self.bert_dropout(bert_encodings)
        # BS X max_nwords x self.bertmodule_outdim
        bert_merged_encodings = pad_sequence(
            [self.get_merged_encodings(bert_seq_encodings, seq_splits, mode='avg') \
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)],
            batch_first=True,
            padding_value=0
        )

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = bert_merged_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # dense
        # [BS,max_nwords,*] or [BS,max_nwords,self.bertmodule_outdim]->[BS,max_nwords,output_dim]
        # logits = self.dense(self.dropout(intermediate_encodings))
        logits = self.dense(intermediate_encodings)

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss
