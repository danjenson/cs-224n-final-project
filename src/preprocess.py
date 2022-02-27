#!/usr/bin/env python3
import argparse
import json
import random
import sys

from bashlint.data_tools import (
    bash_parser,
    ast2template,
)


def preprocess(
    data,
    p_test=0.02,
    p_valid=0.02,
    sep='<|sep|>',
    sep_inv='<|inv|>',
    sep_cmd='<|cmd|>',
    output_train_file=None,
    output_valid_file=None,
    output_test_file=None,
):
    '''
    Steps:
    1. Templatize commands.
    2. Encode data for use by transformer.
    3. Partition into train, validation, and test sets.
    '''
    data = templatize(data)
    data = encode(data, sep, sep_inv, sep_cmd)
    train, valid, test = split(data, p_valid, p_test)
    if output_train_file:
        write(train, output_train_file)
    if output_valid_file:
        write(valid, output_valid_file)
    if output_test_file:
        write(test, output_test_file)
    return train, valid, test


def templatize(data):
    '''Parse commands and replace identifiers with placeholders.'''
    for d in data:
        try:
            d['cmd'] = ast2template(bash_parser(d['cmd']))
        except:
            print(f'unable to templatize: {d["cmd"]}', file=sys.stderr)
            continue
    return data


def encode(data, sep, sep_inv, sep_cmd):
    '''Encode data for use by the transformer.'''
    return [
        f'{sep} {sep_inv} {d["invocation"]} {sep_cmd} {d["cmd"]} {sep}'
        for d in data
    ]


def split(data, p_valid, p_test):
    '''Split rows into train, validation, and test sets.'''
    assert p_valid + p_test < 1, 'no training data!'
    n = len(data)
    n_valid = int(n * p_valid)
    n_test = int(n * p_test)
    random.shuffle(data)
    return (
        data[n_valid + n_test:],
        data[:n_valid],
        data[n_valid:n_valid + n_test],
    )


def write(rows, file):
    '''Writes rows out to the given file.'''
    with open(file, 'w') as f:
        f.write('\n'.join(rows))
    print(f'wrote {file}!', file=sys.stderr)


def parse_args(argv):
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d',
        '--data_json',
        help='''JSON data file with format:
            {1: "invocation": "...", "cmd": "...", 2: ..}
            ''',
        default='nl2bash-data.json',
    )
    parser.add_argument(
        '-p_test',
        help='proportion of data set to be used for testing',
        default=0.02,
    )
    parser.add_argument(
        '-p_valid',
        help='proportion of dataset to be used for validation',
        default=0.02,
    )
    parser.add_argument(
        '-sep',
        help='separator for each <inv, cmd> pair',
        default='<|sep|>',
    )
    parser.add_argument(
        '-sep_inv',
        help='separator for invocations',
        default='<|inv|>',
    )
    parser.add_argument(
        '-sep_cmd',
        help='separator for commands',
        default='<|cmd|>',
    )
    parser.add_argument(
        '-o_train',
        '--output_train_file',
        help='output training data file',
        default='train.txt',
    )
    parser.add_argument(
        '-o_valid',
        '--output_valid_file',
        help='output validation data file',
        default='valid.txt',
    )
    parser.add_argument(
        '-o_test',
        '--output_test_file',
        help='output test data file',
        default='test.txt',
    )
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)
    with open(args.data_json) as f:
        data = json.load(f).values()
    preprocess(
        data,
        args.p_test,
        args.p_valid,
        args.sep,
        args.sep_inv,
        args.sep_cmd,
        args.output_train_file,
        args.output_valid_file,
        args.output_test_file,
    )
