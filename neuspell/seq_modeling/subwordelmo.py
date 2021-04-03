import time

from .evals import get_metrics
from .helpers import *
from .models import SubwordElmo
from .util import is_module_available, get_module_or_attr

"""
NEW: reranking snippets
"""
# (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
import torch
from torch.nn import CrossEntropyLoss

HFACE_batch_size = 8

# from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
# gpt2Tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
# gpt2LMHeadModel = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
# gpt2Tokenizer.add_special_tokens({'pad_token':"[PAD]"})
# gpt2LMHeadModel.resize_token_embeddings(len(gpt2Tokenizer))
# assert gpt2Tokenizer.pad_token == '[PAD]'

from transformers import GPT2Tokenizer, GPT2LMHeadModel

gpt2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
gpt2LMHeadModel = GPT2LMHeadModel.from_pretrained('gpt2-medium')
gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token


# from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
# txlTokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
# txlLMHeadModel = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
# txlTokenizer.pad_token = txlTokenizer.eos_token


def get_losses_from_gpt_lm(this_sents: "list[str]", gpt2LMHeadModel, gpt2Tokenizer, device):
    this_input_ids = gpt2Tokenizer.batch_encode_plus(this_sents, add_special_tokens=True, pad_to_max_length=True,
                                                     add_space_before_punct_symbol=True)["input_ids"]
    this_labels = torch.tensor(
        [[i if i != gpt2Tokenizer.pad_token_id else -100 for i in row] for row in this_input_ids]).to(device)
    this_input_ids = torch.tensor(this_input_ids).to(device)
    this_outputs = gpt2LMHeadModel(input_ids=this_input_ids)
    this_lm_logits = this_outputs[0]
    # Shift so that tokens < n predict n
    shift_logits2 = this_lm_logits[:, :-1, :]
    shift_labels2 = this_labels[:, 1:]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits2.permute(0, 2, 1), shift_labels2)
    losses = loss.sum(dim=-1).cpu().detach().numpy().tolist()

    return losses


def get_losses_from_txl_lm(this_sents: "list[str]", txlLMHeadModel, txlTokenizer, device):
    this_input_ids_dict = txlTokenizer.batch_encode_plus(this_sents, add_special_tokens=True, pad_to_max_length=True,
                                                         add_space_before_punct_symbol=True)
    this_input_ids = this_input_ids_dict["input_ids"]
    chunks = [sum(val) for val in this_input_ids_dict["attention_mask"]]
    chunks_cumsum = np.cumsum(chunks).tolist()

    this_labels = torch.tensor(
        [[i if i != txlTokenizer.pad_token_id else -100 for i in row] for row in this_input_ids]).to(device)
    this_input_ids = torch.tensor(this_input_ids).to(device)
    this_outputs = txlLMHeadModel(input_ids=this_input_ids, labels=this_labels)
    this_loss = this_outputs[0]
    this_loss = this_loss.view(-1).cpu().detach().numpy()
    losses = [sum(this_loss[str_pos:end_pos - 1]) for str_pos, end_pos in zip([0] + chunks_cumsum[:-1], chunks_cumsum)]

    return losses


def load_model(vocab, verbose=False):
    model = SubwordElmo(3 * len(vocab["chartoken2idx"]), vocab["token2idx"][vocab["pad_token"]],
                        len(vocab["token_freq"]))
    if verbose:
        print(model)
    print(get_model_nparams(model))

    return model


def load_pretrained(model, checkpoint_path, optimizer=None, device='cuda'):
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
    print(f"previously, max_dev_acc: {max_dev_acc:.5f} and argmax_dev_acc: {argmax_dev_acc:.5f}")

    if optimizer is not None:
        return model, optimizer, max_dev_acc, argmax_dev_acc

    return model


def model_inference(model, data, topk, device, batch_size=16, beam_search=False, selected_lines_file=None):
    """
    model: an instance of SubwordElmo
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    topk: how many of the topk softmax predictions are considered for metrics calculations
    device: "cuda:0" or "cpu"
    batch_size: batch size for input to the model
    beam_search: if True, greedy topk will not be performed
    """

    if beam_search:
        if topk < 2:
            raise Exception("when using beam_search, topk must be greater than 1, topk is used as beam width")
        else:
            print(f":: doing BEAM SEARCH with topk:{topk} ::")

        if selected_lines_file is not None:
            raise Exception("when using beam_search, ***selected_lines_file*** arg is not used; no implementation")

    # list of dicts with keys {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
    results = []
    line_index = 0

    inference_st_time = time.time()
    VALID_batch_size = batch_size
    valid_loss, valid_acc = 0., 0.
    corr2corr, corr2incorr, incorr2corr, incorr2incorr = 0, 0, 0, 0
    predictions = []
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_batch_size, shuffle=False)
    model.eval()
    model.to(device)
    for batch_id, (batch_clean_sentences, batch_corrupt_sentences) in tqdm(enumerate(data_iter)):
        torch.cuda.empty_cache()
        # st_time = time.time()
        # set batch data
        batch_labels, batch_lengths = labelize(batch_clean_sentences, vocab)
        # batch_idxs, batch_lengths_ = sclstm_tokenize(batch_corrupt_sentences, vocab)
        # assert (batch_lengths_==batch_lengths).all()==True
        # batch_idxs = [batch_idxs_.to(device) for batch_idxs_ in batch_idxs]
        batch_lengths = batch_lengths.to(device)
        batch_labels = batch_labels.to(device)
        elmo_batch_to_ids = get_module_or_attr("allennlp.modules.elmo", "batch_to_ids")
        batch_elmo_inp = elmo_batch_to_ids([line.split() for line in batch_corrupt_sentences]).to(device)
        # forward
        try:
            with torch.no_grad():
                if not beam_search:
                    """
                    NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len) if topk==1
                    """
                    batch_loss, batch_predictions = model(batch_elmo_inp, targets=batch_labels,
                                                          topk=topk)  # topk=1 or 5
                else:
                    """
                    NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk==None
                    """
                    batch_loss, batch_predictions, batch_predictions_probs = model(batch_elmo_inp, targets=batch_labels,
                                                                                   topk=topk, beam_search=True)
        except RuntimeError:
            print(
                f"batch_lengths:{batch_lengths.shape},batch_elmo_inp:{batch_elmo_inp.shape},batch_labels:{batch_labels.shape}")
            raise Exception("")
        valid_loss += batch_loss
        # compute accuracy in numpy
        batch_labels = batch_labels.cpu().detach().numpy()
        batch_lengths = batch_lengths.cpu().detach().numpy()
        # based on beam_search, do either greedy topk or beam search for topk
        if not beam_search:
            # based on topk, obtain either strings of batch_predictions or list of tokens
            if topk == 1:
                batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab,
                                                            batch_corrupt_sentences)
            else:
                batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab,
                                                             batch_corrupt_sentences)
            predictions.extend(batch_predictions)

            batch_clean_sentences = [line.lower() for line in batch_clean_sentences]
            batch_corrupt_sentences = [line.lower() for line in batch_corrupt_sentences]
            batch_predictions = [line.lower() for line in batch_predictions]
            corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
                get_metrics(batch_clean_sentences, batch_corrupt_sentences, batch_predictions, check_until_topk=topk,
                            return_mistakes=False)
            corr2corr += corr2corr_
            corr2incorr += corr2incorr_
            incorr2corr += incorr2corr_
            incorr2incorr += incorr2incorr_

            for i, (a, b, c) in enumerate(zip(batch_clean_sentences, batch_corrupt_sentences, batch_predictions)):
                results.append({"id": line_index + i, "original": a, "noised": b, "predicted": c, "topk": [],
                                "topk_prediction_probs": [], "topk_reranker_losses": []})
            line_index += len(batch_clean_sentences)

        else:
            """
            NEW: use untokenize_without_unks3 for beam search outputs
            """
            # k different lists each of type batch_predictions as in topk==1
            # List[List[Strings]]
            k_batch_predictions, k_batch_predictions_probs = untokenize_without_unks3(batch_predictions,
                                                                                      batch_predictions_probs,
                                                                                      batch_lengths, vocab,
                                                                                      batch_clean_sentences, topk)

            ##########################################################
            ############## this takes top1 as-is #####################
            # corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
            #     get_metrics(batch_clean_sentences,batch_corrupt_sentences,k_batch_predictions[0],check_until_topk=1,return_mistakes=False)
            # corr2corr+=corr2corr_
            # corr2incorr+=corr2incorr_
            # incorr2corr+=incorr2corr_
            # incorr2incorr+=incorr2incorr_

            ##########################################################
            ############### this does reranking ######################
            gpt2LMHeadModel.to(device)
            gpt2LMHeadModel.eval()
            # txlLMHeadModel.to(device)
            # txlLMHeadModel.eval()

            reranked_batch_predictions = []
            batch_clean_sentences_ = []
            batch_corrupt_sentences_ = []
            batch_losses_ = []
            with torch.no_grad():
                for b in range(len(batch_clean_sentences)):
                    losses = []
                    this_sents = [k_batch_predictions[k][b] for k in range(topk)]
                    losses = get_losses_from_gpt_lm(this_sents, gpt2LMHeadModel, gpt2Tokenizer, device)
                    # losses = get_losses_from_txl_lm(this_sents, txlLMHeadModel, txlTokenizer, device)
                    kmin = np.argmin(losses)
                    reranked_batch_predictions.append(k_batch_predictions[kmin][b])
                    batch_clean_sentences_.append(batch_clean_sentences[b])
                    batch_corrupt_sentences_.append(batch_corrupt_sentences[b])
                    batch_losses_.append(losses)

            corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
                get_metrics(batch_clean_sentences_, batch_corrupt_sentences_, reranked_batch_predictions,
                            check_until_topk=1, return_mistakes=False)
            corr2corr += corr2corr_
            corr2incorr += corr2incorr_
            incorr2corr += incorr2corr_
            incorr2incorr += incorr2incorr_

            batch_predictions_k = [[k_batch_predictions[j][i] for j in range(len(k_batch_predictions))] for i in
                                   range(len(k_batch_predictions[0]))]
            batch_predictions_probs_k = [
                [k_batch_predictions_probs[j][i] for j in range(len(k_batch_predictions_probs))] for i in
                range(len(k_batch_predictions_probs[0]))]
            for i, (a, b, c, d, e, f) in \
                    enumerate(zip(batch_clean_sentences_, batch_corrupt_sentences_, reranked_batch_predictions,
                                  batch_predictions_k, batch_predictions_probs_k, batch_losses_)):
                results.append({"id": line_index + i, "original": a, "noised": b, "predicted": c, "topk": d,
                                "topk_prediction_probs": e, "topk_reranker_losses": f})
            line_index += len(batch_clean_sentences)

        # delete
        del batch_loss
        del batch_predictions
        del batch_labels, batch_lengths, batch_elmo_inp
        torch.cuda.empty_cache()

        # '''
        # # update progress
        # progressBar(batch_id+1,
        #             int(np.ceil(len(data) / VALID_batch_size)), 
        #             ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
        #             [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),None,None])
        # '''

    print(f"\nEpoch {None} valid_loss: {valid_loss / (batch_id + 1)}")
    print("total inference time for this data is: {:4f} secs".format(time.time() - inference_st_time))
    print("###############################################")
    print("total token count: {}".format(corr2corr + corr2incorr + incorr2corr + incorr2incorr))
    print(f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
    print(f"accuracy is {(corr2corr + incorr2corr) / (corr2corr + corr2incorr + incorr2corr + incorr2incorr)}")
    print(f"word correction rate is {(incorr2corr) / (incorr2corr + incorr2incorr)}")
    print("###############################################")

    if not beam_search and selected_lines_file is not None:

        print("evaluating only for selected lines ... ")

        assert len(data) == len(predictions), print(len(data), len(predictions), "lengths mismatch")

        if selected_lines_file is not None:
            selected_lines = {num: "" for num in [int(line.strip()) for line in open(selected_lines_file, 'r')]}
        else:
            selected_lines = None

        clean_lines, corrupt_lines, predictions_lines = [tpl[0] for tpl in data], [tpl[1] for tpl in data], predictions

        corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes = \
            get_metrics(clean_lines, corrupt_lines, predictions_lines, return_mistakes=True,
                        selected_lines=selected_lines)

        print("###############################################")
        print("total token count: {}".format(corr2corr + corr2incorr + incorr2corr + incorr2incorr))
        print(
            f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
        print(f"accuracy is {(corr2corr + incorr2corr) / (corr2corr + corr2incorr + incorr2corr + incorr2incorr)}")
        print(f"word correction rate is {(incorr2corr) / (incorr2corr + incorr2incorr)}")
        print("###############################################")

    return results
