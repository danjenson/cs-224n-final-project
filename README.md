# Project Ghost Shell

## Literature Review

- semantic parsing to query languages
- Salesforce semantic parsing
- converting text to code

## Tasks

- data
  - loading
  - augmentation
  - placeholders, i.e. paths, names, etc
    - BART may solve this
    - get me weather in Burma
      - train model to translate to `get me weather in STR1`
      - train model to translate previous sentence to `./get_me_weather.sh --loc STR1`
      - translate back STR1 to Burma
- embeddings
  - loading
  - training
- transformer composition
  - hugging face transformers
    - BART: seq2seq
    - T5: another model
- experimentation framework
  - when to checkpoint models
  - grid search across which parameters/model types?
  - evaluation
    - use existing API/function?

## Questions

- pipes -- composable commands
  - smash together pipes in dataset?
