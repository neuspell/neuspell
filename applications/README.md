# Some applications of `neuspell`

### Defense against adversarial character attacks

We determine the usefulness of our toolkit as a counter-measure against character-level adversarial
attacks [(§ 5.2)](https://arxiv.org/pdf/2010.11085.pdf#page=5). We find that our models are better defenses to
adversarial attacks than previously proposed spell checkers. In this section, we provide a guide to utilize one of the
spell checkers in our toolkit- ```SC-LSTM plus ELMO (at input)``` (aka. `ElmosclstmChecker`) in the setup
of [Adversarial-Misspellings](https://github.com/danishpruthi/Adversarial-Misspellings) ([Pruthi et al. 2019](https://www.aclweb.org/anthology/P19-1561/))
.

```bash
git clone https://github.com/danishpruthi/Adversarial-Misspellings; cd Adversarial-Misspellings
mv ../Adversarial-Misspellings-arxiv/main.py ./
```

And then the following commands can be utilized
- Word level attacks
```bash
CUDA_VISIBLE_DEVICES=x python main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-word-only --attack <swap,drop or key> --num-attacks 2 --model bilstm-word --defense --sc-elmoscrnn --backoff <neutral or pass-through>
CUDA_VISIBLE_DEVICES=x python main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-word-only --attack <add or all> --num-attacks 2 --model bilstm-word --defense --sc-elmoscrnn --small --backoff <neutral or pass-through>
```
- Char level attacks
```bash
CUDA_VISIBLE_DEVICES=x python main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-char-only --attack <swap,drop or key> --num-attacks 2 --model bilstm-char --defense --sc-elmoscrnn --backoff <neutral or pass-through>
CUDA_VISIBLE_DEVICES=x python main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-char-only --attack <add or all> --num-attacks 2 --model bilstm-char --defense --sc-elmoscrnn --small --backoff <neutral or pass-through>
```
- Char+Word level attacks
```bash
CUDA_VISIBLE_DEVICES=x python main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-word-plus-char --attack <swap,drop or key> --num-attacks 2 --model bilstm --defense --sc-elmoscrnn --backoff <neutral or pass-through>
CUDA_VISIBLE_DEVICES=x python main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-word-plus-char --attack <add or all> --num-attacks 2 --model bilstm --defense --sc-elmoscrnn --small --backoff <neutral or pass-through>
```