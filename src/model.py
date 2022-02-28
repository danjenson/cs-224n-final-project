#!/usr/bin/env python3
from types import SimpleNamespace
import argparse
import itertools as it
import json
import multiprocessing as mp
import pathlib
import random
import sys
import yaml

from torch.utils.data.dataset import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
import torch

from bashlint.data_tools import (
    bash_parser,
    ast2template,
)

from evaluation import test_evaluation


class InvalidData(Exception):
    pass


class NL2CMDDataset(Dataset):

    def __init__(self, encoded_command_pairs, tokenizer: PreTrainedTokenizer):
        batch_encoding = tokenizer(encoded_command_pairs,
                                   add_special_tokens=True,
                                   padding=True,
                                   truncation=True,
                                   max_length=100)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def model(yaml_config_path):
    cfg = load_config_yaml(yaml_config_path)
    encoded_command_pairs = preprocess(load_data(cfg.data.sources[0]),
                                       cfg.data.tokens)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.eos_token = cfg.data.tokens.eos
    if not tokenizer.pad_token:
        tokenizer.pad_token = cfg.data.tokens.eos
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train, valid, test = split(encoded_command_pairs, cfg.data.splits)
    train_ds = NL2CMDDataset(train, tokenizer)
    valid_ds = NL2CMDDataset(valid, tokenizer)
    test_ds = NL2CMDDataset(test, tokenizer)
    if pathlib.Path(cfg.model.output_path).is_dir():
        model = AutoModelForCausalLM.from_pretrained(cfg.model.output_path)
#        model.to(torch.device('cuda'))
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**vars(cfg.training)),
            data_collator=collator,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
        )
        trainer.train()
        model.save_pretrained(cfg.model.output_path)
    test_evaluation(test, tokenizer, model, cfg.data.tokens.cmd)


def evaluate(yaml_config_path):
    cfg = load_config_yaml(yaml_config_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.eos_token = cfg.data.tokens.eos
    if not tokenizer.pad_token:
        tokenizer.pad_token = cfg.data.tokens.eos
    model = AutoModelForCausalLM.from_pretrained(cfg.model.output_path)
    test_evaluation(test, tokenizer, model, cfg.data.tokens.cmd)


def load_config_yaml(path):
    '''Load yaml config as a nested namespace object.'''
    with open(path) as f:
        d = yaml.safe_load(f)
    return json.loads(
        json.dumps(d),
        object_hook=lambda d: SimpleNamespace(**d),
    )


def load_data(path):
    '''Dynamically load dataset based on file extension.'''
    ft = file_type(path)
    try:
        d = {
            name.replace('load_data_', ''): func
            for name, func in globals().items()
            if name.startswith('load_data_')
        }[file_type(path)](path)
    except:
        raise InvalidData(f'no data loader for file type: {ft}')
    return d


def file_type(path):
    '''Return a path's file type.'''
    return pathlib.Path(path).suffix.replace('.', '')


def load_data_json(path):
    '''Load json data, returns list of (nl, cmd) pairs.'''
    with open(path) as f:
        data = json.load(f)
    return [
        SimpleNamespace(nl=d['invocation'], cmd=d['cmd'])
        for d in data.values()
    ]


def preprocess(command_pairs, tokens):
    '''Preprocess data.'''
    with mp.Pool() as p:
        return p.starmap(encode, zip(command_pairs, it.repeat(tokens)))


def encode(command_pair, tokens):
    '''Encode data for use by the transformer.'''
    nl, cmd = command_pair.nl, templatize(command_pair.cmd)
    return f'{tokens.eos} {tokens.nl} {nl} {tokens.cmd} {cmd} {tokens.eos}'


def templatize(cmd):
    '''Parse commands and replace identifiers with placeholders.'''
    try:
        return ast2template(bash_parser(cmd))
    except:
        print(f'unable to templatize: {cmd}', file=sys.stderr)
    return cmd


def split(data, splits, seed=0):
    '''Split rows into train, validation, and test sets.'''
    assert splits.train + splits.valid + splits.test == 1, 'invalid data splits!'
    n = len(data)
    n_valid = int(n * splits.valid)
    n_test = int(n * splits.test)
    random.seed(seed)
    random.shuffle(data)
    return (
        data[n_valid + n_test:],
        data[:n_valid],
        data[n_valid:n_valid + n_test],
    )


def parse_args(argv):
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c',
        '--yaml_config_path',
        help='yaml config path',
        default='basic.yaml',
    )
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)
    model(args.yaml_config_path)
