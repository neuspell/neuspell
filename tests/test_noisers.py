from neuspell.noisers import CharacterReplacementNoiser
from neuspell.noisers import ProbabilisticCharacterReplacementNoiser
from neuspell.noisers import WordReplacementNoiser

example_texts = [
    "This is an example sentence to demonstrate noising in the neuspell repository.",
    "Here is another such amazing example!!"
]

noisers = [
    CharacterReplacementNoiser,
    ProbabilisticCharacterReplacementNoiser,
    WordReplacementNoiser,
]

print("\n\n---------------------------------------\n---------------------------------------\n\n")

for noiser in noisers:
    print(f"testing {noiser.__name__}")

    my_noiser = noiser(language="english")
    my_noiser.load_resources()

    # example use-case-1
    noise_texts = my_noiser.noise(example_texts)
    print(noise_texts)

    # example use-case-2
    preprocessor = noiser.create_preprocessor(lower_case=True, remove_accents=True)
    retokenizer = noiser.create_retokenizer()
    noise_texts = my_noiser.noise(example_texts, preprocessor=preprocessor, retokenizer=retokenizer)
    print(noise_texts)

    # example use-case-3
    preprocessor = noiser.create_preprocessor(lower_case=True, remove_accents=True)
    retokenizer = noiser.create_retokenizer(use_spacy_retokenization=True)
    noise_texts = my_noiser.noise(example_texts, preprocessor=preprocessor, retokenizer=retokenizer)
    print(noise_texts)

    print("\n\n---------------------------------------\n---------------------------------------\n\n")
