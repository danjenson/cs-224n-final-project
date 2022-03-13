from pathlib import Path
import json

import pandas as pd
import transformers as hft


class EpochCallback(hft.TrainerCallback):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def on_epoch_end(self, args, state, control, **kwargs):
        self.func(state.epoch)


def build_epoch_predict_callback(
    trainer,
    predict,
    source,
    target,
    output_path,
):
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    preds_path = output_path / 'predictions.json'

    def callback(epoch):
        d = {}
        if preds_path.exists():
            with preds_path.open() as f:
                d = json.load(f)
        d[epoch] = pd.DataFrame({
            'source': trainer.eval_dataset[source],
            'target': trainer.eval_dataset[target],
            'prediction': predict(trainer),
        }).to_dict('records')
        with open(preds_path, 'w') as f:
            json.dump(d, f, indent=2)

    return EpochCallback(callback)
