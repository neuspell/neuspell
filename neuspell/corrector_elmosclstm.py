import os
import time
from typing import List

import numpy as np
import torch

from .commons import spacy_tokenizer, ARXIV_CHECKPOINTS, Corrector
from .seq_modeling.downloads import download_pretrained_model
from .seq_modeling.helpers import load_data, load_vocab_dict, get_model_nparams, sclstm_tokenize, save_vocab_dict
from .seq_modeling.helpers import train_validation_split, batch_iter, labelize, progressBar, batch_accuracy_func
from .seq_modeling.util import is_module_available, get_module_or_attr

if is_module_available("allennlp"):
    from .seq_modeling.elmosclstm import load_model, load_pretrained, model_predictions, model_inference

""" corrector module """


class CorrectorElmoSCLstm(Corrector):

    def __init__(self, tokenize=True, pretrained=False, device="cpu"):
        super(CorrectorElmoSCLstm, self).__init__()

        if not is_module_available("allennlp"):
            raise ImportError(
                "install `allennlp` by running `pip install -r extras-requirements.txt`. See `README.md` for more info.")

        self.tokenize = tokenize
        self.pretrained = pretrained
        self.device = device

        self.ckpt_path = None
        self.vocab_path, self.weights_path = "", ""
        self.model, self.vocab = None, None

        if self.pretrained:
            self.from_pretrained(self.ckpt_path)

    def __model_status(self):
        assert not (self.model is None or self.vocab is None), print("model & vocab must be loaded first")
        return

    def from_pretrained(self, ckpt_path=None, vocab="", weights=""):
        self.ckpt_path = ckpt_path or ARXIV_CHECKPOINTS["elmoscrnn-probwordnoise"]
        self.vocab_path = vocab if vocab else os.path.join(self.ckpt_path, "vocab.pkl")
        if not os.path.isfile(self.vocab_path):  # leads to "FileNotFoundError"
            download_pretrained_model(self.ckpt_path)
        print(f"loading vocab from path:{self.vocab_path}")
        self.vocab = load_vocab_dict(self.vocab_path)
        print(f"initializing model")
        self.model = load_model(self.vocab)
        self.weights_path = weights if weights else self.ckpt_path
        print(f"loading pretrained weights from path:{self.weights_path}")
        self.model = load_pretrained(self.model, self.weights_path, device=self.device)
        return

    def set_device(self, device='cpu'):
        prev_device = self.device
        device = "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        if not (prev_device == device):
            if self.model is not None:
                # please load again, facing issues with just .to(new_device) and new_device
                #   not same the old device, https://tinyurl.com/y57pcjvd
                self.from_pretrained(self.ckpt_path, vocab=self.vocab_path, weights=self.weights_path)
            self.device = device
        print(f"model set to work on {device}")
        return

    def correct(self, x):
        return self.correct_string(x)

    def correct_string(self, mystring: str, return_all=False) -> str:
        x = self.correct_strings([mystring], return_all=return_all)
        if return_all:
            return x[0][0], x[1][0]
        else:
            return x[0]

    def correct_strings(self, mystrings: List[str], return_all=False) -> List[str]:
        self.__model_status()
        if self.tokenize:
            mystrings = [spacy_tokenizer(my_str) for my_str in mystrings]
        data = [(line, line) for line in mystrings]
        batch_size = 4 if self.device == "cpu" else 16
        return_strings = model_predictions(self.model, data, self.vocab, device=self.device, batch_size=batch_size)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

    def correct_from_file(self, src, dest="./clean_version.txt"):
        """
        src = f"{DEFAULT_DATA_PATH}/traintest/corrupt.txt"
        """
        self.__model_status()
        x = [line.strip() for line in open(src, 'r')]
        y = self.correct_strings(x)
        print(f"saving results at: {dest}")
        opfile = open(dest, 'w')
        for line in y:
            opfile.write(line + "\n")
        opfile.close()
        return

    def evaluate(self, clean_file, corrupt_file):
        """
        clean_file = f"{DEFAULT_DATA_PATH}/traintest/clean.txt"
        corrupt_file = f"{DEFAULT_DATA_PATH}/traintest/corrupt.txt"
        """
        self.__model_status()
        batch_size = 4 if self.device == "cpu" else 16
        for x, y, z in zip([""], [clean_file], [corrupt_file]):
            print(x, y, z)
            test_data = load_data(x, y, z)
            _ = model_inference(self.model,
                                test_data,
                                topk=1,
                                device=self.device,
                                batch_size=batch_size,
                                beam_search=False,
                                selected_lines_file=None,
                                vocab_=self.vocab)
        return

    def model_size(self):
        self.__model_status()
        return get_model_nparams(self.model)

    def finetune(self, clean_file, corrupt_file, validation_split=0.2, n_epochs=2, new_vocab_list=[]):

        if new_vocab_list:
            raise NotImplementedError("Do not currently support modifying output vocabulary of the models")

        # load data and split in train-validation
        train_data = load_data("", clean_file, corrupt_file)
        train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
        print("len of train and test data: ", len(train_data), len(valid_data))

        # load vocab and model
        self.__model_status()

        # finetune
        #############################################
        # training and validation
        #############################################
        model, vocab = self.model, self.vocab
        TRAIN_BATCH_SIZE, VALID_BATCH_SIZE = 16, 32
        GRADIENT_ACC = 4
        DEVICE = self.device
        START_EPOCH, N_EPOCHS = 0, n_epochs
        CHECKPOINT_PATH = os.path.join(self.ckpt_path, "finetuned_model")
        VOCAB_PATH = os.path.join(CHECKPOINT_PATH, "vocab.pkl")
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print(f"CHECKPOINT_PATH: {CHECKPOINT_PATH}")

        # running stats
        max_dev_acc, argmax_dev_acc = -1, -1
        patience = 100

        # Create an optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # model to device
        model.to(DEVICE)

        # load parameters if not training from scratch
        if START_EPOCH > 1:
            progress_write_file = open(os.path.join(CHECKPOINT_PATH, f"progress_retrain_from_epoch{START_EPOCH}.txt"),
                                       'w')
            model, optimizer, max_dev_acc, argmax_dev_acc = load_pretrained(model, CHECKPOINT_PATH, optimizer=optimizer)
            progress_write_file.write(f"Training model params after loading from path: {CHECKPOINT_PATH}\n")
        else:
            progress_write_file = open(os.path.join(CHECKPOINT_PATH, "progress.txt"), 'w')
            print(f"Training model params from scratch")
            progress_write_file.write(f"Training model params from scratch\n")
        progress_write_file.flush()

        # train and eval
        for epoch_id in range(START_EPOCH, N_EPOCHS + 1):
            # check for patience
            if (epoch_id - argmax_dev_acc) > patience:
                print("patience count reached. early stopping initiated")
                print("max_dev_acc: {}, argmax_dev_acc: {}".format(max_dev_acc, argmax_dev_acc))
                break
            # print epoch
            print(f"In epoch: {epoch_id}")
            progress_write_file.write(f"In epoch: {epoch_id}\n")
            progress_write_file.flush()
            # train loss and backprop
            train_loss = 0.
            train_acc = 0.
            train_acc_count = 0.
            print("train_data size: {}".format(len(train_data)))
            progress_write_file.write("train_data size: {}\n".format(len(train_data)))
            progress_write_file.flush()
            train_data_iter = batch_iter(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            # for batch_id, (batch_labels,batch_sentences) in tqdm(enumerate(train_data_iter)):
            for batch_id, (batch_labels, batch_sentences) in enumerate(train_data_iter):
                optimizer.zero_grad()
                st_time = time.time()
                # set batch data
                batch_labels, batch_lengths = labelize(batch_labels, vocab)
                batch_idxs, batch_lengths_ = sclstm_tokenize(batch_sentences, vocab)
                assert (batch_lengths_ == batch_lengths).all() == True
                batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
                batch_lengths = batch_lengths.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                elmo_batch_to_ids = get_module_or_attr("allennlp.modules.elmo", "batch_to_ids")
                batch_elmo_inp = elmo_batch_to_ids([line.split() for line in batch_sentences]).to(DEVICE)
                # forward
                model.train()
                loss = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels)
                batch_loss = loss.cpu().detach().numpy()
                train_loss += batch_loss
                # backward
                loss.backward()
                optimizer.step()
                # compute accuracy in numpy
                if batch_id % 10000 == 0:
                    train_acc_count += 1
                    model.eval()
                    with torch.no_grad():
                        _, batch_predictions = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels)
                    model.train()
                    batch_labels = batch_labels.cpu().detach().numpy()
                    batch_lengths = batch_lengths.cpu().detach().numpy()
                    ncorr, ntotal = batch_accuracy_func(batch_predictions, batch_labels, batch_lengths)
                    batch_acc = ncorr / ntotal
                    train_acc += batch_acc
                    # update progress
                progressBar(batch_id + 1,
                            int(np.ceil(len(train_data) / TRAIN_BATCH_SIZE)),
                            ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", "avg_batch_acc"],
                            [time.time() - st_time, batch_loss, train_loss / (batch_id + 1), batch_acc,
                             train_acc / train_acc_count])
                if batch_id == 0 or (batch_id + 1) % 5000 == 0:
                    nb = int(np.ceil(len(train_data) / TRAIN_BATCH_SIZE))
                    progress_write_file.write(f"{batch_id + 1}/{nb}\n")
                    progress_write_file.write(
                        f"batch_time: {time.time() - st_time}, avg_batch_loss: {train_loss / (batch_id + 1)}, avg_batch_acc: {train_acc / (batch_id + 1)}\n")
                    progress_write_file.flush()
            print(f"\nEpoch {epoch_id} train_loss: {train_loss / (batch_id + 1)}")

            try:
                # valid loss
                valid_loss = 0.
                valid_acc = 0.
                print("valid_data size: {}".format(len(valid_data)))
                progress_write_file.write("valid_data size: {}\n".format(len(valid_data)))
                progress_write_file.flush()
                valid_data_iter = batch_iter(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)
                for batch_id, (batch_labels, batch_sentences) in enumerate(valid_data_iter):
                    st_time = time.time()
                    # set batch data
                    batch_labels, batch_lengths = labelize(batch_labels, vocab)
                    batch_idxs, batch_lengths_ = sclstm_tokenize(batch_sentences, vocab)
                    assert (batch_lengths_ == batch_lengths).all() == True
                    batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
                    batch_lengths = batch_lengths.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)
                    elmo_batch_to_ids = get_module_or_attr("allennlp.modules.elmo", "batch_to_ids")
                    batch_elmo_inp = elmo_batch_to_ids([line.split() for line in batch_sentences]).to(DEVICE)
                    # forward
                    model.eval()
                    with torch.no_grad():
                        batch_loss, batch_predictions = model(batch_idxs, batch_lengths, batch_elmo_inp,
                                                              targets=batch_labels)
                    model.train()
                    valid_loss += batch_loss
                    # compute accuracy in numpy
                    batch_labels = batch_labels.cpu().detach().numpy()
                    batch_lengths = batch_lengths.cpu().detach().numpy()
                    ncorr, ntotal = batch_accuracy_func(batch_predictions, batch_labels, batch_lengths)
                    batch_acc = ncorr / ntotal
                    valid_acc += batch_acc
                    # update progress
                    progressBar(batch_id + 1,
                                int(np.ceil(len(valid_data) / VALID_BATCH_SIZE)),
                                ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", "avg_batch_acc"],
                                [time.time() - st_time, batch_loss, valid_loss / (batch_id + 1), batch_acc,
                                 valid_acc / (batch_id + 1)])
                    if batch_id == 0 or (batch_id + 1) % 2000 == 0:
                        nb = int(np.ceil(len(valid_data) / VALID_BATCH_SIZE))
                        progress_write_file.write(f"{batch_id}/{nb}\n")
                        progress_write_file.write(
                            f"batch_time: {time.time() - st_time}, avg_batch_loss: {valid_loss / (batch_id + 1)}, avg_batch_acc: {valid_acc / (batch_id + 1)}\n")
                        progress_write_file.flush()
                print(f"\nEpoch {epoch_id} valid_loss: {valid_loss / (batch_id + 1)}")

                # save model, optimizer and test_predictions if val_acc is improved
                if valid_acc >= max_dev_acc:
                    # to file
                    # name = "model-epoch{}.pth.tar".format(epoch_id)
                    name = "model.pth.tar".format(epoch_id)
                    torch.save({
                        'epoch_id': epoch_id,
                        'previous_max_dev_acc': max_dev_acc,
                        'previous_argmax_dev_acc': argmax_dev_acc,
                        'max_dev_acc': valid_acc,
                        'argmax_dev_acc': epoch_id,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(CHECKPOINT_PATH, name))
                    print("Model saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, name), epoch_id))
                    save_vocab_dict(VOCAB_PATH, vocab)

                    # re-assign
                    max_dev_acc, argmax_dev_acc = valid_acc, epoch_id

            except Exception as e:
                temp_folder = os.path.join(CHECKPOINT_PATH, "temp")
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)
                name = "model.pth.tar".format(epoch_id)
                torch.save({
                    'epoch_id': epoch_id,
                    'previous_max_dev_acc': max_dev_acc,
                    'previous_argmax_dev_acc': argmax_dev_acc,
                    'max_dev_acc': valid_acc,
                    'argmax_dev_acc': epoch_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join(temp_folder, name))
                print("Model saved at {} in epoch {}".format(os.path.join(temp_folder, name), epoch_id))
                save_vocab_dict(VOCAB_PATH, vocab)
                raise Exception(e)

        print(f"Model and logs saved at {os.path.join(CHECKPOINT_PATH, 'model.pth.tar')}")
        return
