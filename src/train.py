#!/usr/bin/env python3
import argparse
import sys

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

# TODO: how do we use?
# 1. DataCollator
# 2. Dataset
# 3. Tokenizer - update from default to include our separators


def train(train_text, model):
    model_config = AutoConfig.from_pretrained(model)
    # TODO: extend tokenizer to add our tokens
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    # TODO: change this to AutoModelForCausalLM?
    model = AutoModelForSeq2SeqLM.from_config(model)
    collator = DataCollatorForSeq2Seq(tokenizer, model)
    train_config = 



def parse_args(argv):
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-t',
        '--train_text',
        help='train data text file',
        default='train.txt',
    )
    parser.add_argument(
        '-m',
        '--model',
        help='https://tinyurl.com/HuggingFaceAutoConfig',
        default='bart',
    )
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)
    train(args.data_text, args.model)
