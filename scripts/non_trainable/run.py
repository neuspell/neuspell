import torch
from candidates_reranking import ContextProbabilityBERTLM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

if __name__ == "__main__":
    contextProbability = ContextProbabilityBERTLM(device=DEVICE)
    tokenized_sentences = [
        ["he", "ran", "in", "the", "prak"],
        ["when", "was", "lest", "tme", "you", "wrte", "to", "me"],
        ["te", "envronment", "is", "geting", "poluted"],
        ["te", "envronment", "is", "geting", "poluted"],
        ["te", "envronment", "is", "geting", "poluted"],
        ["te", "envronment", "is", "geting", "poluted"],
        ["te", "envronment", "is", "geting", "poluted"]
    ]
    indices = [4, 2, 0, 1, 2, 3, 4]
    candidates = [
        ["prka", "mrak", "park", "prank", "peak", "plan", "pork"],
        ["lyst", "lezt", "list", "last", "less", "lust", "lost", "test", "nest", "best"],
        ["t", "we", "me", "th", "the", "teh"],
        ["envros", "envirooonnn", "envos", "environmment", "environment"],
        ["was", "ws", "as", "us"],
        ["geing", "geeting", "getting", "getiing"],
        ["pouuted", "polluted", "pollented"]
    ]
    print(contextProbability._get_bert_probabilities(tokenized_sentences, indices, candidates))
