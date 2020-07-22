


class Candidate:
    def __init__(self, word, prob):
        self.word = word
        self.prob = prob

    def __str__(self):
        return str([self.word, self.prob])

    def __repr__(self):
        return str([self.word, self.prob])
