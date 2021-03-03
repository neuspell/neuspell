import time

from .evals import get_metrics
from .helpers import *
from .models import SCLSTM


def load_model(vocab):
    model = SCLSTM(3 * len(vocab["chartoken2idx"]),
                   vocab["token2idx"][vocab["pad_token"]],
                   len(vocab["token_freq"]))

    print(model)
    print(get_model_nparams(model))

    return model


def load_pretrained(model, checkpoint_path, optimizer=None, device='cpu'):
    if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    print(f"Loading model params from checkpoint dir: {checkpoint_path}")
    checkpoint_data = torch.load(os.path.join(checkpoint_path, "model.pth.tar"), map_location=map_location)
    # print(f"previously model saved at : {checkpoint_data['epoch_id']}")

    model.load_state_dict(checkpoint_data['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    max_dev_acc, argmax_dev_acc = checkpoint_data["max_dev_acc"], checkpoint_data["argmax_dev_acc"]

    if optimizer is not None:
        return model, optimizer, max_dev_acc, argmax_dev_acc

    return model


def model_predictions(model, data, vocab, device, batch_size=16):
    """
    model: an instance of SCLSTM
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    """

    topk = 1
    print("###############################################")
    inference_st_time = time.time()
    final_sentences = []
    VALID_batch_size = batch_size
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_batch_size, shuffle=False)
    model.eval()
    model.to(device)
    for batch_id, (batch_clean_sentences, batch_corrupt_sentences) in enumerate(data_iter):
        # set batch data
        batch_labels, batch_lengths = labelize(batch_clean_sentences, vocab)
        batch_idxs, batch_lengths_ = sclstm_tokenize(batch_corrupt_sentences, vocab)
        assert (batch_lengths_ == batch_lengths).all() == True
        batch_idxs = [batch_idxs_.to(device) for batch_idxs_ in batch_idxs]
        batch_lengths = batch_lengths.to(device)
        batch_labels = batch_labels.to(device)
        # forward
        with torch.no_grad():
            # because topk=1, batch_predictions are of shape (batch_size,batch_max_seq_len)
            _, batch_predictions = model(batch_idxs, batch_lengths, targets=batch_labels, topk=topk)
        batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences)
        final_sentences.extend(batch_predictions)
    print("total inference time for this data is: {:4f} secs".format(time.time() - inference_st_time))
    return final_sentences


def model_inference(model, data, topk, device, batch_size=16, vocab_=None):
    """
    model: an instance of SCLSTM
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    topk: how many of the topk softmax predictions are considered for metrics calculations
    """
    if vocab_ is not None:
        vocab = vocab_
    # list of dicts with keys {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
    results = []
    line_index = 0

    print("###############################################")
    inference_st_time = time.time()
    _corr2corr, _corr2incorr, _incorr2corr, _incorr2incorr = 0, 0, 0, 0
    _mistakes = []
    VALID_batch_size = batch_size
    valid_loss = 0.
    valid_acc = 0.
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_batch_size, shuffle=False)
    model.eval()
    model.to(device)
    for batch_id, (batch_clean_sentences, batch_corrupt_sentences) in tqdm(enumerate(data_iter)):
        st_time = time.time()
        # set batch data
        batch_labels, batch_lengths = labelize(batch_clean_sentences, vocab)
        batch_idxs, batch_lengths_ = sclstm_tokenize(batch_corrupt_sentences, vocab)
        assert (batch_lengths_ == batch_lengths).all() == True
        batch_idxs = [batch_idxs_.to(device) for batch_idxs_ in batch_idxs]
        batch_lengths = batch_lengths.to(device)
        batch_labels = batch_labels.to(device)
        # forward
        with torch.no_grad():
            """
            NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
            """
            batch_loss, batch_predictions = model(batch_idxs, batch_lengths, targets=batch_labels, topk=topk)
        valid_loss += batch_loss
        # compute accuracy in numpy
        batch_labels = batch_labels.cpu().detach().numpy()
        batch_lengths = batch_lengths.cpu().detach().numpy()
        # based on topk, obtain either strings of batch_predictions or list of tokens
        if topk == 1:
            batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab,
                                                        batch_corrupt_sentences)
        else:
            batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab,
                                                         batch_corrupt_sentences, topk=None)

        batch_clean_sentences = [line.lower() for line in batch_clean_sentences]
        batch_corrupt_sentences = [line.lower() for line in batch_corrupt_sentences]
        batch_predictions = [line.lower() for line in batch_predictions]
        corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes = \
            get_metrics(batch_clean_sentences, batch_corrupt_sentences, batch_predictions, check_until_topk=topk,
                        return_mistakes=True)
        _corr2corr += corr2corr
        _corr2incorr += corr2incorr
        _incorr2corr += incorr2corr
        _incorr2incorr += incorr2incorr
        _mistakes.extend(mistakes)

        for i, (a, b, c) in enumerate(zip(batch_clean_sentences, batch_corrupt_sentences, batch_predictions)):
            results.append({"id": line_index + i, "original": a, "noised": b, "predicted": c, "topk": [],
                            "topk_prediction_probs": [], "topk_reranker_losses": []})
        line_index += len(batch_clean_sentences)

        '''
        # update progress
        progressBar(batch_id+1,
                    int(np.ceil(len(data) / VALID_batch_size)), 
                    ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
                    [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),None,None])
        '''
    print(f"\nEpoch {None} valid_loss: {valid_loss / (batch_id + 1)}")
    print("total inference time for this data is: {:4f} secs".format(time.time() - inference_st_time))
    print("###############################################")
    # print("")
    # for mistake in _mistakes:
    #     print(mistake)
    # print("")
    print("total token count: {}".format(_corr2corr + _corr2incorr + _incorr2corr + _incorr2incorr))
    print(
        f"_corr2corr:{_corr2corr}, _corr2incorr:{_corr2incorr}, _incorr2corr:{_incorr2corr}, _incorr2incorr:{_incorr2incorr}")
    print(f"accuracy is {(_corr2corr + _incorr2corr) / (_corr2corr + _corr2incorr + _incorr2corr + _incorr2incorr)}")
    print(f"word correction rate is {(_incorr2corr) / (_incorr2corr + _incorr2incorr)}")
    print("###############################################")
    return results
