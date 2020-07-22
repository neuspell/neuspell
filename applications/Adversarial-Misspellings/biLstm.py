import dynet_config
dynet_config.set(random_seed=42)

import dynet as dy



class BiLSTM():
    def build_model(self, nwords, nchars, ntags):
        self.model = dy.Model()
        trainer = dy.AdamTrainer(self.model)
        EMB_SIZE = 64
        HID_SIZE = 64
        self.W_emb = self.model.add_lookup_parameters((nwords, EMB_SIZE))  # Word embeddings
        self.fwdLSTM = dy.VanillaLSTMBuilder(1, EMB_SIZE, HID_SIZE, self.model)  # Forward RNN
        self.bwdLSTM = dy.VanillaLSTMBuilder(1, EMB_SIZE, HID_SIZE, self.model)  # Backward RNN
        self.W_sm = self.model.add_parameters((ntags, 2 * HID_SIZE))  # Softmax weights
        self.b_sm = self.model.add_parameters((ntags))  # Softmax bias
        return trainer

    # A function to calculate scores for one value
    def calc_scores(self, words, chars):
        dy.renew_cg()
        word_embs = [dy.lookup(self.W_emb, x) for x in words]
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
