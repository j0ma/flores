from collections import defaultdict
import pandas as pd
import numpy as np
import click
import re


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
@click.option('--input_file', required=True,
              help='File to load experimental results from.')
def main(input_file):    
    reported = pd.Series(
        {
            'en-ne': 4.3,
            'ne-en': 7.6,
            'en-si': 1.2,
            'si-en': 7.2
        }
    )

    raw, agg, agg2 = load_seed_results(input_file)

    print('### Raw results')
    print(raw)

    print('\n### Summary statistics')
    print(agg)

    print('\n### Confidence interval')
    print(agg2)

    print('\n### Reported results')
    print(reported)

    print('\n### Difference from reported')
    print(raw - reported)

    print('\n### Fraction of overestimates')
    print(((raw - reported) > 0).mean(axis=0))

if __name__ == '__main__':
    main()
