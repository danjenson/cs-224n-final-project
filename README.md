# Project Ghost Shell

## TODO

- Write predict_one for CLI use
- Test different postprocessing methods
- Data augmentation
- Try non-templated commands
- Beam search with custom eval?
- Different metrics: https://huggingface.co/docs/datasets/metrics

## Example code

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
