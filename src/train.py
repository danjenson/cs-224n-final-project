#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import yaml
from types import SimpleNamespace

import datasets as hfd
import transformers as hft

from bashlint.data_tools import (
    bash_parser,
    ast2template,
)

logging.basicConfig(
    filename='gs.log',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.DEBUG,
)

# TODO:
# 1. compute metric
# 2. do we need a data loader?
# 3. test Causal vs Seq2Seq


def train(cfg):
    '''Train a HuggingFace model.'''
    tokenizer = hft.AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    model = hft.AutoModelForSeq2SeqLM.from_pretrained(cfg.model.checkpoint)
    trans = cfg.dataset.translate
    ds = hfd.load_from_disk(cfg.dataset.path)
    ds = ds.train_test_split(
        test_size=cfg.dataset.splits.test,
        seed=cfg.dataset.splits.seed,
        shuffle=True,
    )
    ds = ds.map(
        lambda x: tokenizer(x[trans.source], x[trans.target], truncation=True),
        batched=True,
    )
    # NOTE: Trainer uses DataCollatorWithPadding by default; requires tokenizer
    trainer = hft.Trainer(
        model=model,
        args=hft.TrainingArguments(**vars(cfg.training)),
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(cfg.model.output_path)


def build_templated_dataset():
    '''Build a templated command HuggingFace Dataset from raw NLC2CMD data.'''
    with open('nl2bash-data.json') as f:
        data = json.load(f)
    d = {'cmd': [], 'nlc': []}
    for v in data.values():
        d['cmd'].append(v['cmd'])
        d['nlc'].append(v['invocation'])
    ds = hfd.Dataset.from_dict(d)
    return ds.map(
        lambda b: {'cmd_templated': [templatize(cmd) for cmd in b["cmd"]]},
        batched=True)


def templatize(cmd):
    '''Parse commands and replace identifiers with placeholders.'''
    try:
        return ast2template(bash_parser(cmd))
    except:
        logging.debug(f'unable to templatize: {cmd}')
    return cmd


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
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d',
        '--dataset',
        help='dataset to use',
        default='tds',
    )
    parser.add_argument(
        '-t',
        '--train_yaml',
        help='train a model with given yaml config',
        default='bart.yaml',
    )
    parser.add_argument(
        '-tds',
        help='build a templated command Dataset',
        action='store_true',
    )
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)
    if args.tds:
        tds = build_templated_dataset()
        tds.save_to_disk('tds')
        logging.info('saved templated dataset to "tds"')
    if args.train_yaml:
        cfg = load_config(args.train_yaml)
        train(cfg)
