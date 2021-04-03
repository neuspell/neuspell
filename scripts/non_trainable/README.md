# candidate_generation_reranking

- Attempt at creating a spell checker based on the concept- generate-candidates-and-rerank. To generate candidates, once
  can leverage edit distance based on characters or sequence-units from double metaphone.
- A typical pipeline explored in non-trainable checkers is to first identify those words in a sentence which are not
  present in a preloaded langauge lexicon. Once such words in a sentence are identified, candidates can be generated for
  those words to be replaced. Upon identifying suitable replacement candidates, one can use a scoring technique to
  narrow the list optionally or directly re-rank the candidates and pick the top-1 as the replacements.
