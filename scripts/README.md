# Readme

- This folder contains scripts for different spell correction systems.
- The folder `trainable` includes neural models for spell correction, trained as sequence labelling task with a prefixed
  dictionary of words as output vocabulary.
- The folder `non_trainable` contains some simple models for candidate-generation-reranking type of spell correction,
  wherein we first select a potential set of replacement words and then rerank them.
- The folder `english_baselines` contains two off-the-shelf spell correctors along their usage.