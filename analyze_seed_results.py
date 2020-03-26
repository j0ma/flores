from collections import defaultdict
import pandas as pd
import re

INPUT_PATH = 'evaluate/seed_results'

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
    agg = raw.describe().loc[['mean', 'std', '25%', '50%', '75%']].T.round(3)
    agg2 = agg[['mean', 'std']].copy()
    agg2['lb'] = agg2['mean'] - 2*agg2['std']
    agg2['ub'] = agg2['mean'] + 2*agg2['std']

    return raw, agg, agg2

if __name__ == '__main__':

    reported = pd.Series(
        {
            'en-ne': 4.3,
            'ne-en': 7.6,
            'en-si': 1.2,
            'si-en': 7.2
        }
    )

    raw, agg, agg2 = load_seed_results(INPUT_PATH)

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
