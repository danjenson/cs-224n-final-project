#!/usr/bin/env python3
import json
import pandas as pd


def main():
    with open('bart.json') as f:
        bart = json.load(f)
    with open('t5.json') as f:
        t5 = json.load(f)
    with open('gpt2.json') as f:
        gpt2 = json.load(f)
    print(
        pd.DataFrame([
            {
                'model': 'GPT2',
                'raw': max(gpt2['scores']['raw']),
                'clean': max(gpt2['scores']['clean']),
                'clean+max_len': max(gpt2['scores']['clean+max_len']),
            },
            {
                'model': 'BART',
                'raw': max(bart['scores']['raw']),
                'clean': max(bart['scores']['clean']),
                'clean+max_len': max(bart['scores']['clean+max_len']),
            },
            {
                'model': 'T5',
                'raw': max(t5['scores']['raw']),
                'clean': max(t5['scores']['clean']),
                'clean+max_len': max(t5['scores']['clean+max_len']),
            },
        ]).to_latex())


if __name__ == '__main__':
    main()
