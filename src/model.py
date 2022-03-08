#!/usr/bin/env python3
import argparse
import json
import logging
import re
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace

import datasets as hfd
import numpy as np
import transformers as hft

from bashlint.data_tools import (
    bash_parser,
    ast2template,
)
import metric_utils

logging.basicConfig(
    filename='gs.log',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.DEBUG,
)


def train(cfg):
    '''Train a HuffingFace model.'''
    trainer = build_trainer(cfg)
    trainer.train()
    p = Path(cfg.output_path)
    trainer.save_model(cfg.output_path)


def build_trainer(cfg, finetuned=False):
    '''Common setup for various subcommands.'''
    d = resolve(cfg.model.task)
    ckpt = cfg.model.checkpoint
    if finetuned:
        ckpt = cfg.output_path
    tokenizer = d['tokenizer'].from_pretrained(ckpt)
    model = d['model'].from_pretrained(ckpt)
    if not finetuned:
        tokenizer, model = tweak(cfg.model.task, tokenizer, model)
    collator = build_collator(
        cfg.model.task,
        d['collator'],
        tokenizer,
        model,
    )
    trans = cfg.dataset.translate
    ds = hfd.load_from_disk(cfg.dataset.path)
    f = lambda x: d['tokenize'](tokenizer, x, trans.source, trans.target)
    if cfg.model.task == 'causal':
        ds['train'] = ds['train'].map(f, batched=True)
        f = lambda x: d['tokenize'](
            tokenizer, x, trans.source, trans.target, is_test=True)
        ds['test'] = ds['test'].map(f, batched=True)
    else:
        ds = ds.map(f, batched=True)
    return d['trainer'](
        model=model,
        args=d['args'](**vars(cfg.training)),
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        data_collator=collator,
    )


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


def tweak(task, tokenizer, model):
    '''Tweak model and tokenizer for specific task.'''
    if task == 'causal':
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<|source|>', '<|target|>']})
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def build_collator(task, cls, tokenizer, model):
    '''Build an appropriate data collator.'''
    if task == 'causal':
        return cls(tokenizer, mlm=False)
    return cls(tokenizer, model)


def tokenize_seq2seq(tokenizer, examples, source, target):
    '''Tokenize for a seq2seq model.'''
    inputs = tokenizer(examples[source], truncation=True)
    with tokenizer.as_target_tokenizer():
        inputs['labels'] = tokenizer(examples[target],
                                     truncation=True)['input_ids']
    return inputs


def tokenize_causal(tokenizer, examples, source, target, is_test=False):
    '''Tokenize for a causal model.'''
    t = SimpleNamespace(
        bos=tokenizer.bos_token,
        eos=tokenizer.eos_token,
        src='<|source|>',
        dst='<|target|>',
    )

    if is_test:
        f = lambda a: f'{t.bos} {t.src} {a} {t.dst}'
        encoded = list(map(f, examples[source]))
    else:
        f = lambda a, b: f'{t.bos} {t.src} {a} {t.dst} {b} {t.eos}'
        encoded = list(map(f, examples[source], examples[target]))
    return tokenizer(encoded, truncation=True)


def predict(cfg):
    '''Evaluate a model given a config.'''
    trainer = build_trainer(cfg, finetuned=True)
    ds = trainer.eval_dataset
    predict = predict_seq2seq
    if cfg.model.task == 'causal':
        predict = predict_causal
    preds = predict(trainer, ds)
    return ds.add_column('pred', preds)


def predict_seq2seq(trainer, ds):
    '''Predict using seq2seq LM.'''
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.predict
    pad_id = -100
    res = trainer.predict(ds)
    decode = trainer.tokenizer.decode
    return [
        decode(p[np.where(p != pad_id)], skip_special_tokens=True)
        for p in res.label_ids
    ]


def predict_causal(trainer, ds):
    '''Predict using causal LM.'''
    eos = trainer.tokenizer.eos_token_id
    loader = trainer.get_eval_dataloader(ds)
    decode = trainer.tokenizer.decode
    preds = []
    for batch in loader:
        ps = trainer.model.generate(
            input_ids=batch['input_ids'].clone().detach().cuda(),
            max_length=100,
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
        )
        preds.extend([decode(p) for p in ps])
    return preds


def score(cfg, postprocess_funcs=[]):
    '''Score output using optional postprocessing function.'''
    path = Path(cfg.output_path) / 'preds'
    ds = hfd.load_from_disk(path)
    # it seems that only the last func from the list will be in effect
    for func in postprocess_funcs:
        ds = ds.map(func)

    def score(example):
        example['score'] = metric_utils.compute_metric(
            example['pred'], 1.0, example[cfg.dataset.translate.target])
        return example

    return np.mean(ds.map(score)['score'])


def clean(example):
    '''Clean up causal predictions.'''
    example['pred'] = re.sub(
        ' +',
        ' ',
        example['pred'].split('<|target|>')[-1].strip(),
    )
    return example


def max_len(example):
    '''Limit length of prediction.'''
    # TODO(Yingxiao): refactor `post_process`
    special_char = "|"
    max_words = 15;
    input = example['pred'].split()
    if len(input) == 0:
        return None
    output = [input[0]]
    counter = 0
    for i in range(len(input) - 1):
        if input[i + 1] != input[i]:
            if input[i + 1] == special_char:
                counter += 1
                if counter >= 4:
                    break
            output.append(input[i + 1])
            if(input[i+1][-1] == ";"):
                break
    #truncate string
    if len(output) > max_words:
        output = output[:max_words]
    example['pred'] = " ".join(output)
    return example


def build_templated_dataset(p_test=0.02, seed=0):
    '''Build a templated command HuggingFace Dataset from raw NLC2CMD data.'''
    with open('nl2bash-data.json') as f:
        data = json.load(f)
    d = {'cmd': [], 'nlc': []}
    for v in data.values():
        d['cmd'].append(v['cmd'])
        d['nlc'].append(v['invocation'])
    ds = hfd.Dataset.from_dict(d)
    ds = ds.map(
        lambda b: {'cmd_templated': [templatize(cmd) for cmd in b["cmd"]]},
        batched=True)
    return ds.train_test_split(test_size=p_test, seed=seed, shuffle=True)


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
    cls = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=argv[0], formatter_class=cls)
    sub = parser.add_subparsers(dest='command')
    dataset = sub.add_parser('dataset', formatter_class=cls)
    train = sub.add_parser('train', formatter_class=cls)
    predict = sub.add_parser('predict', formatter_class=cls)
    score = sub.add_parser('score', formatter_class=cls)
    dataset.add_argument(
        '-t',
        '--dataset_type',
        help='type of dataset to create',
        default='templated',
        choices=['templated'],
    )
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
    predict.add_argument(
        '-c',
        '--config',
        help='yaml config to use',
        default='bart.yaml',
    )
    score.add_argument(
        '-c',
        '--config',
        help='yaml config to use',
        default='bart.yaml',
    )
    score.add_argument(
        '-fs',
        '--postprocessing_funcs',
        help='postprocessing function names',
        default=['clean'],
    )
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)
    cmd = args.command
    if cmd == 'dataset':
        tds = build_templated_dataset(args.p_test, args.seed)
        tds.save_to_disk(args.output_path)
        print(f'saved to {args.output_path}')
    elif cmd == 'train':
        cfg = load_config(args.config)
        train(cfg)
    elif cmd == 'predict':
        cfg = load_config(args.config)
        preds = predict(cfg)
        path = Path(cfg.output_path) / 'preds'
        preds.save_to_disk(path)
        print(f'saved to {path}')
    elif cmd == 'score':
        cfg = load_config(args.config)
        funcs = None
        if args.postprocessing_funcs:
            funcs = [globals()[f] for f in args.postprocessing_funcs]
        s = score(cfg, funcs)
        print(f'Score: {s}')
