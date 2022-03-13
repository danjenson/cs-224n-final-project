import json

import pandas as pd
import transformers as hft


class EpochCallback(hft.TrainerCallback):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def on_epoch_end(self, **kwargs):
        self.func(kwargs['state'].epoch)


def build_epoch_predict_callback(
    trainer,
    predict,
    source,
    target,
    output_path,
):

    def callback(epoch):
        d = {}
        if output_path.exists():
            with open(output_path) as f:
                d = json.load(f)
        d[epoch] = pd.DataFrame({
            'source': trainer.eval_dataset[source],
            'target': trainer.eval_dataset[target],
            'prediction': predict(trainer),
        }).to_dict('records')
        with open(output_path, 'w') as f:
            json.dump(d, f, indent=2)

    return EpochCallback(callback)
