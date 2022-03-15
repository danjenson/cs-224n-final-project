#!/usr/bin/env python3
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')


def main():
    with open('bart.json') as f:
        bart = json.load(f)
    with open('t5.json') as f:
        t5 = json.load(f)
    with open('gpt2.json') as f:
        gpt2 = json.load(f)
    bart_loss = loss_df(bart)
    t5_loss = loss_df(t5)
    gpt2_loss = loss_df(gpt2)
    sns.lineplot(x=bart_loss.epoch, y=bart_loss.loss, color='blue')
    sns.lineplot(x=gpt2_loss.epoch, y=gpt2_loss.loss, color='orange')
    sns.lineplot(x=t5_loss.epoch, y=bart_loss.loss, color='green')
    plt.legend(labels=['Bart', 'GPT2', 'T5'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.savefig('loss.png')
    plt.clf()
    both_df = pd.DataFrame({
        'bart': bart['scores']['clean+max_len'],
        't5': t5['scores']['clean+max_len'],
        'gpt2': gpt2['scores']['clean+max_len'],
    })
    clean_df = pd.DataFrame({
        'bart': bart['scores']['clean'],
        't5': t5['scores']['clean'],
        'gpt2': gpt2['scores']['clean'],
    })
    sns.lineplot(x=both_df.index + 1, y=both_df.bart, color='blue')
    sns.lineplot(x=both_df.index + 1, y=both_df.gpt2, color='orange')
    sns.lineplot(x=both_df.index + 1, y=both_df.t5, color='green')
    sns.lineplot(x=clean_df.index + 1,
                 y=clean_df.bart,
                 color='blue',
                 linestyle='--')
    sns.lineplot(x=clean_df.index + 1,
                 y=clean_df.gpt2,
                 color='orange',
                 linestyle='--')
    sns.lineplot(x=clean_df.index + 1,
                 y=clean_df.t5,
                 color='green',
                 linestyle='--')
    plt.axhline(y=-0.19, color='red')
    plt.title('Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('NLC2CMD Score')
    plt.legend(
        labels=[
            'Bart: Clean+MaxLen',
            'GPT2: Clean+MaxLen',
            'T5: Clean+MaxLen',
            'Bart: Clean',
            'GPT2: Clean',
            'T5: Clean',
            'GPT3 Baseline',
        ],
        loc='center right',
        bbox_to_anchor=(1, 0.525),
        prop={'size': 8},
    )
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.savefig('metric.png')
    plt.clf()


def loss_df(data):
    return pd.DataFrame({
        'loss': [x['loss'] for x in data['loss'] if x['epoch'] <= 10],
        'epoch': [x['epoch'] for x in data['loss'] if x['epoch'] <= 10],
    })


if __name__ == '__main__':
    main()
