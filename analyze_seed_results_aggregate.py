# Analyzes all seed experiment results
# Jonne Saleva, 2020

from analyze_seed_results import load_seed_results
import pandas as pd
import click
import glob
import os

def process(r, bpe):
    r = r.stack().reset_index()
    r.columns = ['seed', 'language_pair', 'bleu']
    r['bpe'] = bpe
    return r

@click.command()
@click.option('--input_pattern', default='evaluate/seed-experiment-results/raw/seed-results-bpe*')
@click.option('--output_file')
@click.option('--save', is_flag=True, default=False)
def main(input_pattern, output_file, save):
    FILES = glob.glob(input_pattern)
    BPE_SIZES = [int(f[-4:]) for f in FILES]
    results = [load_seed_results(f)[0] for f in FILES]
    results = [process(r, bpe) for r, bpe in zip(results, BPE_SIZES)]
    aggregate = pd.concat(results).sort_values(['bpe', 'seed'])
    print(aggregate)
    if save:
        aggregate.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
