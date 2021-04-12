# Noising Resources

A folder consisting of resources required for various noisers

### Mapping between the folder names and the noising classes

| Folder                          | Class name                                | Disk space (approx.) |
|---------------------------------|-------------------------------------------|----------------------|
| `en-word-replacement-noise`     | `WordReplacementNoiser`                   | 2 MB                 |
| `en-char-replacement-noise`     | `CharacterReplacementNoiser`              | --                   |
| `en-probchar-replacement-noise` | `ProbabilisticCharacterReplacementNoiser` | 80 MB                |

### Usage

```python
from neuspell.noising import WordReplacementNoiser

example_texts = [
    "This is an example sentence to demonstrate noising in the neuspell repository.",
    "Here is another such amazing example !!"
]

word_repl_noiser = WordReplacementNoiser(language="english")
word_repl_noiser.load_resources()
noise_texts = word_repl_noiser.noise(example_texts)
print(noise_texts)
```