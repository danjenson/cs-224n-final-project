# Project Ghost Shell

## Literature Review

- semantic parsing to query languages
- Salesforce semantic parsing
- converting text to code
- https://towardsdatascience.com/building-a-python-code-generator-4b476eec5804
- [translation](https://github.com/huggingface/transformers/blob/master/examples/pytorch/translation/run_translation.py)

## Tasks

1. Get data

- modularize this

2. Repurpose code:

- extract parts:
  - `config.py`
  - `data_utils.py`
  - `generate.py`
  - `preprocess.py`
  - `run.py`
  - `tune.py`

3. Steps:

- ensure using gpu
- load data
- process data
- score predictions
- load BART

- [starter code](https://github.com/nokia/nlc2cmd-submission-hubris)

  - `notebooks`:
    - `data_preprocess.ipynb`: adds additional data sources
    - `graphs.ipynb`: visualize results from experiments
    - `old`: directory of other ipynb that we're ignoring...
  - `src`:
    - `config.py`: configuration used by
    - `data_utils.py`: reading/saving/context/encode/decode from competition
    - `dataset.py`: subclassing torch datasets; LBL vs Blocked dataset?
    - `diverse_beam_search.py`: expanding hugging face beam search
    - `generate.py`: prediction and scoring utilities
    - `modified_beam_search.py`: another modified hugging face beam search
    - `onnx.py`: conversion to/from onnx ML format
    - `preprocess.py`: preprocesses data
    - `run.py`: main loop for training
    - `trainer.py`: overrides hugging face trainer to override some options
    - `tune.py`: contains a list of experiments by modifying config
  - `webapp`
  - `demo_app.py`: starts a flask server
  - `eval.py`: used for predictions using the model on hugging face
  - `requirements.txt`: python requirements
  - `train.py`: runs `experiments()` from `src/tune.py`

# Old Tasks

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
