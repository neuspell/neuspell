# eval scripts

#####################################
# USAGE
# -----
# simple usage
# -----
# python evals.py ../../data/traintest/test.bea4k ../../data/traintest/test.bea4k.noise ./gec-pseudodata/test.bea4k.predictions.txt ./gec-pseudodata test.bea4k.lines.txt 
# python evals.py ../../data/traintest/test.bea60k ../../data/traintest/test.bea60k.noise ./gec-pseudodata/test.bea60k.predictions.txt ./gec-pseudodata test.bea60k.lines.txt
# python evals.py ../../data/traintest/test.bea20k ../../data/traintest/test.bea20k.noise ./gec-pseudodata/test.bea20k.predictions.txt ./gec-pseudodata test.bea20k.lines.txt
# python evals.py ../../data/traintest/test.jfleg ../../data/traintest/test.jfleg.noise ./gec-pseudodata/test.jfleg.predictions.txt ./gec-pseudodata test.jfleg.lines.txt
# python evals.py ../../data/traintest/test.1blm ../../data/traintest/test.1blm.noise.prob ./gec-pseudodata/test.1blm.noise.prob.predictions.txt ./gec-pseudodata test.1blm.noise.prob.lines.txt
# python evals.py ../../data/traintest/test.1blm ../../data/traintest/test.1blm.noise.word ./gec-pseudodata/test.1blm.noise.word.predictions.txt ./gec-pseudodata test.1blm.word.prob.lines.txt
# -----
# usage w/ selected lines
# -----
# python evals.py ../../data/traintest/test.bea4k ../../data/traintest/test.bea4k.noise ./gec-pseudodata/test.bea4k.predictions.txt ./gec-pseudodata test.bea4k.lines.txt ./gec-pseudodata/test.bea4k.lines.txt
# python evals.py ../../data/traintest/test.bea60k ../../data/traintest/test.bea60k.noise ./gec-pseudodata/test.bea60k.predictions.txt ./gec-pseudodata test.bea60k.lines.txt ./gec-pseudodata/test.bea60k.lines.txt
# python evals.py ../../data/traintest/test.bea20k ../../data/traintest/test.bea20k.noise ./gec-pseudodata/test.bea20k.predictions.txt ./gec-pseudodata test.bea20k.lines.txt ./gec-pseudodata/test.bea20k.lines.txt
# python evals.py ../../data/traintest/test.jfleg ../../data/traintest/test.jfleg.noise ./gec-pseudodata/test.jfleg.predictions.txt ./gec-pseudodata test.jfleg.lines.txt ./gec-pseudodata/test.jfleg.lines.txt
#####################################

# from tqdm import tqdm

def __load_data(path):
    opfile = open(path, "r")
    data = opfile.readlines()
    opfile.close()
    return data


def __get_data(inp):
    if isinstance(inp, list):
        data = inp
    elif isinstance(inp, str):
        data = __load_data(inp)
    else:
        raise Exception("arg inp must be either clean_data or a path to fetch clean_data")
    return data


def get_metrics(clean, corrupt, predictions, check_until_topk=1, return_mistakes=False, window=7,
                return_selected_lines=False, selected_lines=None):
    """
    clean: 
        a list of clean sentences, same number of sentences as batch size
        a path to obtain the lines of clean sentences
    corrupt: 
        a list of corrupt sentences, same number of sentences as batch size
        a path to obtain the lines of clean sentences
    predictions:
        is a list of list of lists or
        is a list[list[str]] or 2D numpy array (each of which are tokens from vocab)
            first dimension being batch size
            the second being for ntokens in that sentence 
            the third being for 1 or topk prediction(s) of words
        is a list of sentences
            first dimension being batch size
            the second being for ntokens in that sentence
    check_until_topk:
        compute accuracy, etc. by checking topk words
    """

    clean_data = __get_data(clean)
    corrupt_data = __get_data(corrupt)
    assert len(clean_data) == len(corrupt_data) == len(predictions)

    if isinstance(predictions[0], str):
        predictions = [line.split() for line in predictions]  # to list[list[str]]

    is_correct_prediction = None
    if isinstance(predictions[0][0], str):
        is_correct_prediction = lambda clean_token, corrupt_token: clean_token == corrupt_token
    elif isinstance(predictions[0][0], list):
        is_correct_prediction = lambda clean_token, preds_list: clean_token in preds_list[:check_until_topk]
    else:
        raise Exception("invalid format for predictions")

    if return_mistakes:
        mistakes = []
        # mistakes.append( ("clean_token", "corrupt_token","prediction_tokens","corrupt_context") )

    corr2corr, corr2incorr, incorr2corr, incorr2incorr = 0, 0, 0, 0
    nlines = []
    if selected_lines is not None:
        assert isinstance(selected_lines, dict), print(f"{type(selected_lines)} typefound when expecting type dict")
        print(f"evaluating only for selected lines: {len(selected_lines)}/{len(clean_data)}")
    for i, (clean_line, corrupt_line, predictions_) in enumerate(zip(clean_data, corrupt_data, predictions)):
        if selected_lines is not None:
            if not i in selected_lines:
                continue

        clean_line_tokens, corrupt_line_tokens = clean_line.split(), corrupt_line.split()
        assert len(clean_line_tokens) == len(corrupt_line_tokens)

        # predictions_ can be of a list of shape (len(clean_line_tokens),topk) or a list of (len(clean_line_tokens),) tokens
        if return_selected_lines:
            if len(clean_line_tokens) != len(predictions_):
                continue
        predictions_ = predictions_[:len(clean_line_tokens)]

        nlines.append(i)
        for i, (clean_token, corrupt_token, prediction_tokens) in enumerate(
                zip(clean_line_tokens, corrupt_line_tokens, predictions_)):
            if clean_token == corrupt_token and is_correct_prediction(clean_token, prediction_tokens):
                corr2corr += 1
            elif clean_token == corrupt_token and not is_correct_prediction(clean_token, prediction_tokens):
                corr2incorr += 1
            elif clean_token != corrupt_token and is_correct_prediction(clean_token, prediction_tokens):
                incorr2corr += 1
            elif clean_token != corrupt_token and not is_correct_prediction(clean_token, prediction_tokens):
                incorr2incorr += 1
                if return_mistakes: \
                        mistakes.append((clean_token,
                                         corrupt_token,
                                         prediction_tokens,
                                         " ".join(corrupt_line_tokens[
                                                  max(i - window, 0):min(i + window + 1, len(corrupt_line_tokens))]))
                                        )

    if return_selected_lines:
        print(f"#lines evaluated: {len(nlines)}/{len(clean_data)}")
        if return_mistakes:
            return corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes, nlines
        else:
            return corr2corr, corr2incorr, incorr2corr, incorr2incorr, nlines
    else:
        if return_mistakes:
            return corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes
        else:
            return corr2corr, corr2incorr, incorr2corr, incorr2incorr


if __name__ == "__main__":

    # NEW
    ####################################################
    # Notes
    # It is observed that some sentences predicted by gec models are not 
    #   of same length as input sentences. So, Such sentences can be 
    #   ignored from evaluation.
    # First, get such sentences using ```return_selected_lines``` arg
    # Second, evaluate for only those sentences by passing ```selected_lines``` path
    ####################################################

    import sys

    clean_file_path = sys.argv[1]
    corrupt_file_path = sys.argv[2]
    predictions_file_path = sys.argv[3]
    evaluated_line_nums_savedir = sys.argv[4]
    evaluated_line_nums_savename = sys.argv[5]
    if len(sys.argv) > 6:
        selected_lines_file = sys.argv[6]  # only evaluate for a selected set of line numbers in the predictions
    else:
        selected_lines_file = None

    clean_lines = []
    opfile = open(clean_file_path, 'r')
    for line in opfile: clean_lines.append(line.strip())
    opfile.close()

    corrupt_lines = []
    opfile = open(corrupt_file_path, 'r')
    for line in opfile: corrupt_lines.append(line.strip())
    opfile.close()

    predictions_lines = []
    opfile = open(predictions_file_path, 'r')
    for line in opfile: predictions_lines.append(line.strip())
    opfile.close()

    print(len(clean_lines), len(corrupt_lines), len(predictions_lines))
    assert len(clean_lines) == len(corrupt_lines) == len(predictions_lines)

    if selected_lines_file is not None:
        selected_lines = {num: "" for num in [int(line.strip()) for line in open(selected_lines_file, 'r')]}
        corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes = \
            get_metrics(clean_lines, corrupt_lines, predictions_lines, return_mistakes=True,
                        selected_lines=selected_lines)
    else:
        print(f"selected_lines_file: {selected_lines_file}")
        corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes, nlines = \
            get_metrics(clean_lines, corrupt_lines, predictions_lines, return_mistakes=True, return_selected_lines=True)
        import os

        if not os.path.exists(evaluated_line_nums_savedir):
            os.makedirs(evaluated_line_nums_savedir)
        opfle = open(os.path.join(evaluated_line_nums_savedir, evaluated_line_nums_savename), "w")
        for num in nlines:
            opfle.write(f"{num}\n")
        opfle.close()

    print("")
    print("")
    print("total token count: {}".format(corr2corr + corr2incorr + incorr2corr + incorr2incorr))
    print(f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
    print(f"accuracy is {(corr2corr + incorr2corr) / (corr2corr + corr2incorr + incorr2corr + incorr2incorr)}")
    print(f"word correction rate is {(incorr2corr) / (incorr2corr + incorr2incorr)}")

    # print("")
    # for mistake in mistakes:
    #     print(mistake)

'''
def get_metrics2(clean,corrupt,predictions,return_mistakes=False,window=7):
    """
    predictions: is a list of list of lists; the first dimesnion being batch size, 
        the second being for ntokens in that sentence and 
        the third being for topk predictions
    clean: a list of clean sentences
    corrupt: a list of corrupt sentences
    """
    clean_data = __get_data(clean)
    corrupt_data = __get_data(corrupt)
    assert len(clean_data)==len(corrupt_data)==len(predictions)

    if return_mistakes: mistakes = []

    corr2corr, corr2incorr, incorr2corr, incorr2incorr = 0, 0, 0, 0
    for clean_line,corrupt_line,predictions_ in zip(clean_data,corrupt_data,predictions):
        clean_line_tokens, corrupt_line_tokens = clean_line.split(),corrupt_line.split()
        assert len(clean_line_tokens)==len(corrupt_line_tokens)

        # predictions_ can be of a list of shape (len(clean_line_tokens),topk)
        predictions_ = predictions_[:len(clean_line_tokens)]

        for i, (clean_token,corrupt_token,prediction_tokens) in enumerate( zip(clean_line_tokens, corrupt_line_tokens, predictions_) ):
            if clean_token==corrupt_token and clean_token in prediction_tokens:
                corr2corr+=1
            elif clean_token==corrupt_token and clean_token not in prediction_tokens:
                corr2incorr+=1
            elif clean_token!=corrupt_token and clean_token in prediction_tokens:
                incorr2corr+=1
            elif clean_token!=corrupt_token and clean_token not in prediction_tokens:
                incorr2incorr+=1
                #if return_mistakes: mistakes.append( (" ".join(clean_line_tokens[max(i-window,0):min(i+window+1,len(clean_line_tokens))]),
                #                                      corrupt_token,
                #                                      prediction_tokens) )
                if return_mistakes: mistakes.append( (clean_token,corrupt_token,prediction_tokens,
                                                      " ".join(corrupt_line_tokens[max(i-window,0):min(i+window+1,len(corrupt_line_tokens))])) )
    print("")
    print("total token count: {}".format(corr2corr+corr2incorr+incorr2corr+incorr2incorr))
    print(f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
    print(f"accuracy is {(corr2corr+incorr2corr)/(corr2corr+corr2incorr+incorr2corr+incorr2incorr)}")
    print(f"word correction rate is {(incorr2corr)/(incorr2corr+incorr2incorr)}")

    if return_mistakes: return corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes
    
    return corr2corr, corr2incorr, incorr2corr, incorr2incorr
'''

'''
def get_metrics(clean,corrupt,prediction,return_mistakes=False):
    clean_data = __get_data(clean)
    corrupt_data = __get_data(corrupt)
    prediction_data = __get_data(prediction)
    assert len(clean_data)==len(corrupt_data)==len(prediction_data)

    if return_mistakes: mistakes = []
    window = 5

    corr2corr, corr2incorr, incorr2corr, incorr2incorr = 0, 0, 0, 0
    for clean_line,corrupt_line,prediction_line in tqdm(zip(clean_data,corrupt_data,prediction_data)):
        clean_line_tokens, corrupt_line_tokens, prediction_line_tokens = clean_line.split(),corrupt_line.split(),prediction_line.split()
        for i, (clean_token,corrupt_token,prediction_token) in enumerate( zip(clean_line_tokens, corrupt_line_tokens, prediction_line_tokens) ):
            if clean_token==corrupt_token and prediction_token==clean_token:
                corr2corr+=1
            elif clean_token==corrupt_token and prediction_token!=clean_token:
                corr2incorr+=1
            elif clean_token!=corrupt_token and prediction_token==clean_token:
                incorr2corr+=1
            elif clean_token!=corrupt_token and prediction_token!=clean_token:
                incorr2incorr+=1
                if return_mistakes: mistakes.append( (" ".join(clean_line_tokens[max(i-window,0):min(i+window+1,len(clean_line_tokens))]),
                                                      corrupt_token,
                                                      prediction_token) )

    print("")
    print("total token count: {}".format(corr2corr+corr2incorr+incorr2corr+incorr2incorr))
    print(f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
    print(f"accuracy is {(corr2corr+incorr2corr)/(corr2corr+corr2incorr+incorr2corr+incorr2incorr)}")
    print(f"word correction rate is {(incorr2corr)/(incorr2corr+incorr2incorr)}")

    if return_mistakes: return corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes

    return corr2corr, corr2incorr, incorr2corr, incorr2incorr
'''
