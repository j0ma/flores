from collections import defaultdict
import pandas as pd
import numpy as np
import click
import os
import re

BLEU_REGEX='version.* = (\d+\.\d+)'
PAIRS = ('en-ne', 'ne-en', 'en-si', 'si-en')
SEEDS = (10, 11, 12, 13, 14)

def get_bleu(report):
    return float(report.split(' = ')[1][:4])

def load_seed_results_sacrebleu(p="./translation-output/"):
    methods = os.listdir(p)
    results = []
    for method in methods:
        for seed in SEEDS:
            for pair in PAIRS:
                src, tgt = pair.split('-')
                fname = f"{p}/{method}/seed-{seed}/{pair}.output.raw.log"
                try:
                    with open(fname, 'r') as f:
                        bleu_report = f.readlines()[-1]
                except FileNotFoundError:
                    continue

                if not bleu_report:
                    continue

                bleu_score = get_bleu(bleu_report)
                results.append({
                    'pair': pair,
                    'method': method,
                    'seed': seed,
                    'bleu': bleu_score
                })
    raw = pd.DataFrame(results)
    agg = raw.groupby(['method', 'pair']).bleu.describe().copy()
    agg2 = agg[['mean', 'std']].copy()
    agg = agg[['count','mean', 'std', '25%', '50%', '75%']]
    raw = raw.set_index(['method','seed', 'pair']).unstack()
    agg2['std_err'] = agg2['std']/np.sqrt(agg['count'].copy())
    agg2['lb'] = agg2['mean'] - 2*agg2['std_err']
    agg2['ub'] = agg2['mean'] + 2*agg2['std_err']
    agg = agg.round(3)
    agg2 = agg2.round(3)

    return raw, agg, agg2


def load_seed_results(p):
    results = defaultdict(list)
    for line in open(p, 'r'):
        if line.strip() == '':
            continue
        elif line.startswith("="):
            continue
        elif line.startswith('seed'):
            results['seed'].append(int(line.replace('seed=', '')))
        else:
            lang_pair, result = line.split(' | ')
            results[lang_pair].append(float(result))

    raw = pd.DataFrame(results).set_index('seed')
    agg = raw.describe().loc[['count','mean', 'std', '25%', '50%', '75%']].T.round(3)
    agg2 = agg[['mean', 'std']].copy()
    agg2['std_err'] = agg2['std']/np.sqrt(agg['count'].copy())
    agg2['lb'] = agg2['mean'] - 2*agg2['std_err']
    agg2['ub'] = agg2['mean'] + 2*agg2['std_err']
    agg = agg.round(3)
    agg2 = agg2.round(3)

    return raw, agg, agg2

@click.command()
@click.option('--input_file', required=False,
              help='File to load experimental results from.')
@click.option('--translation_output', required=False,
              help='Folder to load translation output & BLEUsfrom.')
@click.option('--output_file', required=False)
def main(input_file=None, translation_output=None, output_file=None):    
    legacy_mode = bool(input_file is not None) 
    if legacy_mode: print('Legacy mode activated')
    reported = pd.Series(
        {
            'en-ne': 4.3,
            'ne-en': 7.6,
            'en-si': 1.2,
            'si-en': 7.2
        }
    )

    if legacy_mode:
        raw, agg, agg2 = load_seed_results(input_file)
    else:
        raw, agg, agg2 = load_seed_results_sacrebleu(translation_output)

    print('### Raw results')
    print(raw)#.unstack().stack(0).reset_index(0))

    print('\n### Summary statistics')
    print(agg)

    print('\n### Confidence interval')
    print(agg2)

    print('\n### Reported results from paper')
    print(reported)

    # TODO: figure out a fast way to do this diff
    #       when multiple models are present
    if legacy_mode:
        print('\n### Difference from reported')
        print(raw - reported)

        print('\n### Fraction of scores above paper')
        print(((raw - reported) > 0).mean(axis=0))

    if output_file:
        print("Outputting to CSV...")
        raw.stack().reset_index().to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
