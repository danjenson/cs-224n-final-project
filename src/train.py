#!/usr/bin/env python3
import argparse
import itertools as it
import json
import logging
import multiprocessing as mp
import sys
import yaml
from types import SimpleNamespace

import torch
import datasets as hfd
import transformers as hft

from bashlint.data_tools import (
    bash_parser,
    ast2template,
)
from metric_utils import compute_metric

logging.basicConfig(
    filename='gs.log',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.DEBUG,
)

# TODO:
# 1. Test tokenization / causal LM
# 2. Write evaluate
# 3. Test GPT-2 vs. BART vs. T-5
# 4. Try non-templated commands
# 5. Visualization / errors

# 1. Different postprocessing
# 2. Data augmentation


def train(cfg):
    '''Train a HuffingFace model.'''
    d = resolve(cfg.model.task)
    tokenizer = d['tokenizer'].from_pretrained(cfg.model.checkpoint)
    model = d['model'].from_pretrained(cfg.model.checkpoint)
    tokenizer, model = tune(cfg.model.task, tokenizer, model)
    trans = cfg.dataset.translate
    ds = hfd.load_from_disk(cfg.dataset.path)
    ds = ds.train_test_split(
        test_size=cfg.dataset.splits.test,
        seed=cfg.dataset.splits.seed,
        shuffle=True,
    )
    ds = ds.map(lambda x: d['tokenize']
                (tokenizer, x, trans.source, trans.target))
    trainer = d['trainer'](
        model=model,
        args=d['args'](**vars(cfg.training)),
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        data_collator=d['collator'](tokenizer, model),
    )
    trainer.train()
    model.save_pretrained(cfg.model.output_path)


def resolve(task):
    '''Resolve task specific classes and functions.'''
    return {
        'seq2seq': {
            'tokenizer': hft.AutoTokenizer,
            'model': hft.AutoModelForSeq2SeqLM,
            'collator': hft.DataCollatorForSeq2Seq,
            'trainer': hft.Seq2SeqTrainer,
            'args': hft.Seq2SeqTrainingArguments,
            'tokenize': tokenize_seq2seq,
        },
        'causal': {
            'tokenizer': hft.AutoTokenizer,
            'model': hft.AutoModelForCausalLM,
            'collator': hft.DataCollatorForLanguageModeling,
            'trainer': hft.Trainer,
            'args': hft.TrainingArguments,
            'tokenize': tokenize_causal,
        },
    }[task]


def tune(task, tokenizer, model):
    if task == 'causal':
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<|source|>', '<|target|>']})
        model.resize_token_embeddings(len(tokenizer))
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def tokenize_seq2seq(tokenizer, examples, source, target):
    '''Tokenize for a seq2seq model.'''
    inputs = tokenizer(examples[source], truncation=True)
    with tokenizer.as_target_tokenizer():
        inputs['labels'] = tokenizer(examples[target],
                                     truncation=True)['input_ids']
    return inputs


def tokenize_causal(tokenizer, examples, source, target):
    '''Tokenize for a causal model.'''
    t = SimpleNamespace(
        bos=tokenizer.bos_token,
        eos=tokenizer.eos_token,
        src='<|source|>',
        dst='<|target|>',
    )

    def encode(example):
        a, b = example[source], example[target]
        return f'{t.bos} {t.src} {a} {t.dst} {b} {t.eos}'

    return tokenizer(map(encode, examples), truncation=True)


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


def evaluate(cfg):
    '''Evaluate a model given a config.'''
    d = resolve(cfg.model.task)
    tokenizer = d['tokenizer'].from_pretrained(cfg.model.checkpoint)
    model = d['model'].from_pretrained(cfg.model.output_path)
    model.to(torch.device('cuda'))
    # TODO(yingxiao)
    raise NotImplementedError()


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
        '--train',
        help='train a model with given yaml config',
        default='bart.yaml',
    )
    parser.add_argument(
        '-tds',
        help='build a templated command Dataset',
        action='store_true',
    )
    parser.add_argument(
        '-e',
        '--evaluate',
        help='evaluate a model with a given yaml config',
        default='bart.yaml',
    )
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)
    if args.tds:
        tds = build_templated_dataset()
        tds.save_to_disk('tds')
        logging.info('saved templated dataset to "tds"')
    if args.train:
        cfg = load_config(args.train)
        train(cfg)
    if args.evaluate:
        cfg = load_config(args.evaluate)
        evaluate(cfg)
