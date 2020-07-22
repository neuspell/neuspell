birkbeck_data = "../../data/missp.dat"

def read_birkbeck():
    """Returns pairs of (incorrect, correct) spellings."""
    misspellings = []
    with open(birkbeck_data) as f:
        correct = None
        incorrect = []
        for line in f:
            word = line.strip()
            if word.startswith("$"):
                # new group
                if correct is not None:
                    misspellings += [(correct, ii) for ii in incorrect]
                    incorrect = []
                correct = word[1:]
            else:
                incorrect += [word]
    return misspellings
