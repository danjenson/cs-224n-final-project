#!/usr/bin/env python3
from importlib import import_module
from types import SimpleNamespace
from pathlib import Path
import argparse
import json
import logging
import yaml
import sys

import pandas as pd

logging.basicConfig(
    filename='gs.log',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.DEBUG,
)


def dataset(args):
    from dataset import build_templated_dataset
    ds = build_templated_dataset(args.p_test, args.seed)
    ds.save_to_disk(args.output_path)
    print(f'saved to {args.output_path}')


def train(args):
    import trainer_callbacks
    cfg = load_config(args.config)
    task = import_module(cfg.model.task)
    trainer = task.build_trainer(cfg)
    trainer.add_callback(
        trainer_callbacks.build_epoch_predict_callback(
            trainer,
            task.predict,
            cfg.dataset.translate.source,
            cfg.dataset.translate.target,
            Path(cfg.output_path) / 'predictions.json',
        ))
    trainer.train()
    trainer.save_model(cfg.output_path)
    print(f'saved model to {cfg.output_path}')


def score(args):
    import postprocessing as pp
    import metric_utils as mu
    with open(args.predictions_path) as f:
        d = json.load(f)
    pp_funcs = [getattr(pp, func) for func in args.postprocessing_funcs]
    scores = {}
    for epoch, data in d.items():
        df = pd.DataFrame(data)
        for pp_func in pp_funcs:
            df['prediction'] = df['prediction'].apply(pp_func)
        df['score'] = df.apply(
            lambda row: mu.compute_metric(
                row['prediction'],
                1.0,
                row['target'],
            ),
            axis=1,
        )
        scores[epoch] = df['score'].mean()
    print(json.dumps(scores, indent=2))


def load_config(yaml_path):
    '''Load yaml config as a nested namespace object.'''
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    return json.loads(
        json.dumps(d),
        object_hook=lambda d: SimpleNamespace(**d),
    )


def parse_args(argv):
    '''Parse command line arguments.'''
    cls = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=argv[0], formatter_class=cls)
    sub = parser.add_subparsers(dest='command')
    dataset = sub.add_parser('dataset', formatter_class=cls)
    train = sub.add_parser('train', formatter_class=cls)
    score = sub.add_parser('score', formatter_class=cls)
    dataset.add_argument(
        '-p_test',
        help='proportion of dataset to use for testing',
        default=0.02,
    )
    dataset.add_argument(
        '-s',
        '--seed',
        help='seed to be used when shuffling before train-test split',
        default=0,
    )
    dataset.add_argument(
        '-o',
        '--output_path',
        help='output path for dataset',
        default='tds',
    )
    train.add_argument(
        '-c',
        '--config',
        help='yaml config to use',
        default='bart.yaml',
    )
    score.add_argument(
        'predictions_path',
        help='path to predictions JSON file',
    )
    score.add_argument(
        '-fs',
        '--postprocessing_funcs',
        help='postprocessing function names',
        default=[],
    )
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)
    globals()[args.command](args)
