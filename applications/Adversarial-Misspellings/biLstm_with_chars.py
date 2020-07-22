import dynet as dy


class BiLSTM():
    def build_model(self, nwords, nchars, ntags):
        self.model = dy.Model()
        trainer = dy.AdamTrainer(self.model)
        EMB_SIZE = 64
        HID_SIZE = 128 
        CHAR_EMB_SIZE = 32
        CHAR_HID_SIZE = 32

        self.W_emb = self.model.add_lookup_parameters((nwords, EMB_SIZE))  # Word embeddings
        self.C_emb = self.model.add_lookup_parameters((nchars, CHAR_EMB_SIZE)) # Char embeddings

        self.char_fwdLSTM = dy.VanillaLSTMBuilder(1, CHAR_EMB_SIZE, CHAR_HID_SIZE, self.model)
        self.char_bwdLSTM = dy.VanillaLSTMBuilder(1, CHAR_EMB_SIZE, CHAR_HID_SIZE, self.model)

        self.fwdLSTM = dy.VanillaLSTMBuilder(1, 2*CHAR_HID_SIZE + EMB_SIZE, HID_SIZE, self.model)  # Forward RNN
        self.bwdLSTM = dy.VanillaLSTMBuilder(1, 2*CHAR_HID_SIZE + EMB_SIZE, HID_SIZE, self.model)  # Backward RNN
        self.W_sm = self.model.add_parameters((ntags, 2 * HID_SIZE))  # Softmax weights
        self.b_sm = self.model.add_parameters((ntags))  # Softmax bias
        return trainer

    
    def get_char_embeddings(self, word):
        # word is a list a character indices
        char_embs = [dy.lookup(self.C_emb, x) for x in word]
        char_fwd_init = self.char_fwdLSTM.initial_state()
        char_fwd_embs = char_fwd_init.transduce(char_embs)
        char_bwd_init = self.char_bwdLSTM.initial_state()
        char_bwd_embs = char_bwd_init.transduce(reversed(char_embs))
        return dy.concatenate([char_fwd_embs[-1], char_bwd_embs[-1]])
        
    # A function to calculate scores for one value
    def calc_scores(self, words, chars):
        dy.renew_cg()
        word_embs = [dy.concatenate([dy.lookup(self.W_emb, words[x]), self.get_char_embeddings(chars[x])]) for x in range(len(words))]
        fwd_init = self.fwdLSTM.initial_state()
        fwd_embs = fwd_init.transduce(word_embs)
        bwd_init = self.bwdLSTM.initial_state()
        bwd_embs = bwd_init.transduce(reversed(word_embs))
        W_sm_exp = dy.parameter(self.W_sm)
        b_sm_exp = dy.parameter(self.b_sm)
        return W_sm_exp * dy.concatenate([fwd_embs[-1], bwd_embs[-1]]) + b_sm_exp

    def load(self, model_file):
        self.model.populate(model_file)
        return

    def save(self, model_file):
        self.model.save(model_file)
        return
