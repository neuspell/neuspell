## Combating Adversarial Misspellings


Code for the following paper. 

> Combating Adversarial Misspellings with Robust Word Recognition
> 
> *Danish Pruthi, Bhuwan Dhingra and Zachary C. Lipton*
> 
> The 57th Annual Meeting of the Association for Computational Linguistics (ACL-19) (To Appear).


### Requirements

```
nltk
dynet
allennlp
torchvision
```

As we use NLTK stopwords, you might have to download stopwords

```
$ python3.6
  >>> import nltk
  >>> nltk.download('stopwords')
```

## Reproducing Results 

### Attacks (Table 3)

You can attack the already trained BiLSTM (word-only, char-only or word+char) models using swap/drop/key-board/add attacks. To do so use the following command.

```
CUDA_VISIBLE_DEVICES=0 python3.6   main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-word-only --attack swap --num-attacks 2 --model 
bilstm-char
```

Upon successfully running this, you would be able to see how a BiLSTM word only model (with an initial test accuracy of 79.19%) is reduced to 63.9% and 53.6% on one and two character swap attacks respectively.

You can change the arguments:

`--attack`: the choice of attack. Available options: `swap`, `drop`, `add`, `key`, and `all` (for exhaustively trying all types of attacks)

```--load```: load the pre-trained BiLSTM model. Available options: `bilstm-char-only`, `bilstm-word-only`,  `bilstm-word-plus-char`. Note that you would have to select the appropriate option in the `--model` flag below.

`--model`: the type of the model. Use `bilstm-word`, `bilstm-char` for word-only and char-only models. If you wish to use the BiLSTM word+char model, use only `bilstm`. 

`--small`: optionally, you can use the small flag to test only on a fraction of the test set (200 examples).

**TODO**: add pre-trained BERT models.

### Defenses (Table 3)

```
CUDA_VISIBLE_DEVICES python3.6 main.py --dynet-seed 1 --mode dev --load model_dumps/bilstm-word-only --attack swap --num-attacks 2 --model bilstm-word --defense
```

Upon successfully running this, you would notice that the accuracy is restored to 78+% on 1/2 character swap attacks.

The only difference from the above is the `--defense` flag. The default defense is a semi-character based RNN (ScRNN) which reconstructs word with a backoff strategy of passing through words when predicted as `UNK` (ScRNN + Pass-Through). If you wish to run ScRNN with other backoff strategies use the following flags

`--defense --sc-neutral`: for ScRNN + Neutral

`--defense --sc-background` for ScRNN + Background

#### Baseline Defense

We use `ATD` spell-checker as our baseline defense. 
Make sure you have the ATD package downloaded and running as
a server on localhost.

Download the source and models from [here](http://www.polishmywriting.com/download/atd_distribution081310.tgz)

Then follow the installation / test instructions [here](https://open.afterthedeadline.com/how-to/get-started/)

This will run the ATD server on localhost at 127.0.0.1:1049. Once this is done, you can use the `--defense --sc-atd` flag to run the ATD spell checker.


## Standalone Usage

You can also use (and train) ScRNN as a defense independently for your use case. To directly use a pretrained ScRNN + PassThrough defense, do the following:

```
$ cd defenses/scRNN/
$ python3.6


>> from corrector import ScRNNChecker
>> checker = ScRNNChecker()
>> checker.correct_string("nicset atcing I have ever witsesed")
'nicest acting i have ever witnessed'
```

The above defense was trained on the Stanford Sentiment Treebank (SST corpus).

**TODO**: add instructions to train defenses on your own corpus.  
 
### Note

If you use the code, please consider citing:

```
@article{pruthi2019combating,
  title={Combating Adversarial Misspellings with Robust Word Recognition},
  author={Pruthi, Danish and Dhingra, Bhuwan and Lipton, Zachary C},
  booktitle = {The 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  address = {Florence, Italy},
  month = {July},
  year = {2019}
}
```