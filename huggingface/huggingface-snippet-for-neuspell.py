import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification


def load_vocab_dict(path_: str):
    """
    path_: path where the vocab pickle file is saved
    """
    with open(path_, 'rb') as fp:
        vocab = pickle.load(fp)
    return vocab


def _tokenize_untokenize(input_text: str, bert_tokenizer):
    subtokens = bert_tokenizer.tokenize(input_text)
    output = []
    for subt in subtokens:
        if subt.startswith("##"):
            output[-1] += subt[2:]
        else:
            output.append(subt)
    return " ".join(output)


def _custom_bert_tokenize_sentence(input_text, bert_tokenizer, max_len):
    tokens = []
    split_sizes = []
    text = []
    for token in _tokenize_untokenize(input_text, bert_tokenizer).split(" "):
        word_tokens = bert_tokenizer.tokenize(token)
        if len(tokens) + len(word_tokens) > max_len - 2:  # 512-2 = 510
            break
        if len(word_tokens) == 0:
            continue
        tokens.extend(word_tokens)
        split_sizes.append(len(word_tokens))
        text.append(token)

    return " ".join(text), tokens, split_sizes


def _custom_bert_tokenize(batch_sentences, bert_tokenizer, padding_idx=None, max_len=512):
    if padding_idx is None:
        padding_idx = bert_tokenizer.pad_token_id

    out = [_custom_bert_tokenize_sentence(text, bert_tokenizer, max_len) for text in batch_sentences]
    batch_sentences, batch_tokens, batch_splits = list(zip(*out))
    batch_encoded_dicts = [bert_tokenizer.encode_plus(tokens) for tokens in batch_tokens]
    batch_input_ids = pad_sequence(
        [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=padding_idx)
    batch_attention_masks = pad_sequence(
        [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=0)
    batch_bert_dict = {"attention_mask": batch_attention_masks,
                       "input_ids": batch_input_ids
                       }
    return batch_sentences, batch_bert_dict, batch_splits


def _custom_get_merged_encodings(bert_seq_encodings, seq_splits, mode='avg', keep_terminals=False, device="cpu"):
    bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
    bert_cls_enc = bert_seq_encodings[0:1, :]
    bert_sep_enc = bert_seq_encodings[-1:, :]
    bert_seq_encodings = bert_seq_encodings[1:-1, :]
    # a tuple of tensors
    split_encoding = torch.split(bert_seq_encodings, seq_splits, dim=0)
    batched_encodings = pad_sequence(split_encoding, batch_first=True, padding_value=0)
    if mode == 'avg':
        seq_splits = torch.tensor(seq_splits).reshape(-1, 1).to(device)
        out = torch.div(torch.sum(batched_encodings, dim=1), seq_splits)
    elif mode == "add":
        out = torch.sum(batched_encodings, dim=1)
    elif mode == "first":
        out = batched_encodings[:, 0, :]
    else:
        raise Exception("Not Implemented")

    if keep_terminals:
        out = torch.cat((bert_cls_enc, out, bert_sep_enc), dim=0)
    return out


if __name__ == "__main__":
    path = "murali1996/bert-base-cased-spell-correction"
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    bert_model = AutoModelForTokenClassification.from_pretrained(path, config=config)
    model_dict = bert_model.state_dict()

    bert_model.eval()
    with torch.no_grad():
        misspelled_sentences = ["Well,becuz badd spelln is ard to undrstnd wen ou rid it.",
                                "they fought a deadly waer",
                                "Hurahh!! we mad it...."]
        batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(misspelled_sentences, tokenizer)
        # print(batch_sentences, "\n")
        outputs = bert_model(batch_bert_dict['input_ids'], attention_mask=batch_bert_dict["attention_mask"],
                             output_hidden_states=True)
        sequence_output = outputs[1][-1]
        # sanity check -------->
        # sequence_output = bert_model.dropout(sequence_output)
        # temp_logits = bert_model.classifier(sequence_output)
        # x1 = [val.data for val in outputs[0].reshape(-1,)]
        # x2 = [val.data for val in temp_logits.reshape(-1,)]
        # assert all([a == b for a, b in zip(x1, x2)])
        # <-------- sanity check
        bert_encodings_splitted = \
            [_custom_get_merged_encodings(bert_seq_encodings, seq_splits, mode='avg')
             for bert_seq_encodings, seq_splits in zip(sequence_output, batch_splits)]
        bert_merged_encodings = pad_sequence(bert_encodings_splitted,
                                             batch_first=True,
                                             padding_value=0
                                             )  # [BS,max_nwords_without_cls_sep,768]
        logits = bert_model.classifier(bert_merged_encodings)
        output_vocab = load_vocab_dict("vocab.pkl")
        # print(logits.shape)
        assert len(output_vocab["idx2token"]) == logits.shape[-1]
        argmax_inds = torch.argmax(logits, dim=-1)
        outputs = [" ".join([output_vocab["idx2token"][idx.item()] for idx in argmaxs][:len(wordsplits)])
                   for wordsplits, argmaxs in zip(batch_splits, argmax_inds)]
        print(outputs)

        print("complete")
