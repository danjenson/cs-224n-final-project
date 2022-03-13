import json
import logging

import datasets as hfd

from bashlint.data_tools import (
    bash_parser,
    ast2template,
)


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
