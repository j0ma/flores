from bleurt import score
import pandas as pd
import click
import sys

def read_lines(fp, lowercase=True):
    with open(fp, "r") as f:
        lines = [line.lower().strip() if lowercase else line.strip() for line in f.readlines()]

        return lines

# @click.command()
# @click.option("--hypotheses-file", required=True)
# @click.option("--references-file", required=True)
# @click.option("--output-file")
# @click.option("--bleurt-checkpoint", default="bleurt-base-128")
# def main(
    # hypotheses_file,
    # references_file,
    # output_file,
    # bleurt_checkpoint
# ):
def main():
    # hypotheses_file = "./translation-output-wmt19-additional/lmvr-tuned/seed-10/kk-en.output.raw.stitched.detok.en" 
    hypotheses_file = "./translation-output/subword-nmt/seed-10/ne-en.output.raw.stitched.detok.en"
    # references_file = "./data/wmt19-kk-additional/kk-en/interim/test/test.kken.en"
    references_file = "./data/wiki_ne_en_bpe5000_lowercase/test.en"
    bleurt_checkpoint = "./bleurt-base-128"

    hypotheses = read_lines(hypotheses_file)
    references = read_lines(references_file)
    references_truecase = read_lines(references_file, lowercase=False)

    scorer = score.BleurtScorer(bleurt_checkpoint)
    scores = scorer.score(references, hypotheses)
    scores = pd.Series(scores)
    max_ix = scores.argmax()
    print(f"Reference for best: {references_truecase[max_ix]}")
    print(f"Best translation: {hypotheses[max_ix]}")
    # assert type(scores) == list and len(scores) == 1
    print(scores.describe())

if __name__ == "__main__":
    main()
