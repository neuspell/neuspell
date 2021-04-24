import os
import time
from typing import List

import numpy as np
import torch

from .commons import spacy_tokenizer, DEFAULT_TRAINTEST_DATA_PATH
from .corrector import Corrector
from .seq_modeling.helpers import load_data, sclstm_tokenize, save_vocab_dict
from .seq_modeling.helpers import train_validation_split, batch_iter, labelize, progressBar, batch_accuracy_func
from .util import is_module_available, get_module_or_attr

if is_module_available("allennlp"):
    from .seq_modeling.elmosclstm import load_model, load_pretrained, model_predictions, model_inference

""" corrector module """


class ElmosclstmChecker(Corrector):

    def __init__(self, **kwargs):

        if not is_module_available("allennlp"):
            raise ImportError(
                "install `allennlp` by running `pip install -r extras-requirements.txt`. "
                "See `README.md` for more info.")

        super().__init__(**kwargs)

    def load_model(self, ckpt_path):
        print(f"initializing model")
        initialized_model = load_model(self.vocab)
        self.model = load_pretrained(initialized_model, self.ckpt_path, device=self.device)

    def correct_strings(self, mystrings: List[str], return_all=False) -> List[str]:
        self.is_model_ready()
        if self.tokenize:
            mystrings = [spacy_tokenizer(my_str) for my_str in mystrings]
        data = [(line, line) for line in mystrings]
        batch_size = 4 if self.device == "cpu" else 16
        return_strings = model_predictions(self.model, data, self.vocab, device=self.device, batch_size=batch_size)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

    def evaluate(self, clean_file, corrupt_file, data_dir=""):
        self.is_model_ready()
        data_dir = DEFAULT_TRAINTEST_DATA_PATH if data_dir == "default" else data_dir

        batch_size = 4 if self.device == "cpu" else 16
        for x, y, z in zip([data_dir], [clean_file], [corrupt_file]):
            print(x, y, z)
            test_data = load_data(x, y, z)
            _ = model_inference(self.model,
                                test_data,
                                topk=1,
                                device=self.device,
                                batch_size=batch_size,
                                vocab_=self.vocab)
        return

    def finetune(self, clean_file, corrupt_file, data_dir="", validation_split=0.2, n_epochs=2,
                 new_vocab_list: List = None):

        if new_vocab_list:
            raise NotImplementedError("Do not currently support modifying output vocabulary of the models")

        # load data and split in train-validation
        data_dir = DEFAULT_TRAINTEST_DATA_PATH if data_dir == "default" else data_dir
        train_data = load_data(data_dir, clean_file, corrupt_file)
        train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
        print("len of train and test data: ", len(train_data), len(valid_data))

        # load vocab and model
        self.is_model_ready()

        # finetune
        #############################################
        # training and validation
        #############################################
        model, vocab = self.model, self.vocab
        TRAIN_BATCH_SIZE, VALID_BATCH_SIZE = 16, 32
        GRADIENT_ACC = 4
        DEVICE = self.device
        START_EPOCH, N_EPOCHS = 0, n_epochs
        CHECKPOINT_PATH = os.path.join(self.ckpt_path if self.ckpt_path else data_dir, "new_models",
                                       os.path.split(self.bert_pretrained_name_or_path)[-1])
        if os.path.exists(CHECKPOINT_PATH):
            num = 1
            while True:
                NEW_CHECKPOINT_PATH = CHECKPOINT_PATH + f"-{num}"
                if not os.path.exists(NEW_CHECKPOINT_PATH):
                    break
                num += 1
            CHECKPOINT_PATH = NEW_CHECKPOINT_PATH
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
                # batch_lengths = batch_lengths.to(device)
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
                    # batch_lengths = batch_lengths.to(device)
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
