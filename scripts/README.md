
# Readme

- This folder contains scripts for different spell correction systems.
- The folder `seq_modeling` includes neural models for spell correction, trained as sequence labelling task with a prefixed dictionary of words as output vocabulary.
- The folder `candidate_generation_reranking` contains some simple models for candidate-generation-reranking type of spell correction, wherein we first select a potential set of replacement words and then rerank them.
- The folder `baselines` contains two off-the-shelf spell correctors along their usage.