from collections import defaultdict
import pandas as pd
import numpy as np
import click
import os

PAIRS = ("en-ne", "ne-en", "en-si", "si-en", "kk-en", "en-kk", "gu-en", "en-gu")
SEEDS = range(10, 15)
METRICS = ('BLEU', 'CHRF3')


def get_bleu(report):
    return float(report.split(" = ")[1][:4])


def get_chrf3(report):
    return float(report.replace("ChrF3 = ", "").strip())


def load_seed_results_sacrebleu(p):
    methods = [m for m in os.listdir(p) if not m.endswith(".py")]
    results = []

    for method in methods:
        for seed in SEEDS:
            for pair in PAIRS:
                src, tgt = pair.split("-")
                bleu_fname = (
                    f"{p}/{method}/seed-{seed}/{pair}.output.raw.bleu.log"
                )
                chrf3_fname = (
                    f"{p}/{method}/seed-{seed}/{pair}.output.raw.chrf3.log"
                )
                try:
                    with open(bleu_fname, "r") as f:
                        bleu_report = f.readlines()[-1]
                    with open(chrf3_fname, "r") as f:
                        chrf3_report = f.readlines()[-1]
                except FileNotFoundError:
                    continue

                if not bleu_report:
                    continue

                bleu_score = get_bleu(bleu_report)
                chrf3_score = get_chrf3(chrf3_report)
                results.append(
                    {
                        "pair": pair,
                        "method": method,
                        "seed": seed,
                        "bleu": bleu_score,
                        "chrf3": chrf3_score,
                    }
                )
    raw = pd.DataFrame(results)
    agg = raw.groupby(["pair", "method"])[['bleu', 'chrf3']].describe().copy()
    agg_bleu = agg.bleu.copy()
    agg_chrf3 = agg.chrf3.copy()
    results = {"raw": raw}
    for metric, agg in (('bleu', agg_bleu), ('chrf3', agg_chrf3)):
        agg2 = agg[["mean", "std"]].copy()
        agg = agg[["count", "mean", "std", "25%", "50%", "75%"]]
        # raw = raw.set_index(["method", "seed", "pair"]).unstack()
        agg2["std_err"] = agg2["std"] / np.sqrt(agg["count"].copy())
        agg2["lb"] = agg2["mean"] - 2 * agg2["std_err"]
        agg2["ub"] = agg2["mean"] + 2 * agg2["std_err"]
        agg = agg.round(3)
        agg2 = agg2.round(3)
        results[metric] = (agg, agg2)

    return results


def load_seed_results(p):
    results = defaultdict(list)

    for line in open(p, "r"):
        if line.strip() == "":
            continue
        elif line.startswith("="):
            continue
        elif line.startswith("seed"):
            results["seed"].append(int(line.replace("seed=", "")))
        else:
            lang_pair, result = line.split(" | ")
            results[lang_pair].append(float(result))

    raw = pd.DataFrame(results).set_index("seed")
    agg = (
        raw.describe()
        .loc[["count", "mean", "std", "25%", "50%", "75%"]]
        .T.round(3)
    )
    agg2 = agg[["mean", "std"]].copy()
    agg2["std_err"] = agg2["std"] / np.sqrt(agg["count"].copy())
    agg2["lb"] = agg2["mean"] - 2 * agg2["std_err"]
    agg2["ub"] = agg2["mean"] + 2 * agg2["std_err"]
    agg = agg.round(3)
    agg2 = agg2.round(3)

    return raw, agg, agg2


@click.command()
@click.option(
    "--input_file",
    required=False,
    help="File to load experimental results from.",
)
@click.option(
    "--translation_output",
    required=False,
    help="Folder to load translation output & BLEUsfrom.",
)
@click.option("--output_file", required=False)
def main(input_file=None, translation_output=None, output_file=None):
    legacy_mode = bool(input_file is not None)

    if legacy_mode:
        print("Legacy mode activated")

    if legacy_mode:
        raw, agg, agg2 = load_seed_results(input_file)
    else:
        agg_results = load_seed_results_sacrebleu(translation_output)
        raw = agg_results['raw']

    print("### Raw results")
    print(raw)
    
    if legacy_mode:
        print("\n### Summary statistics - BLEU")
        print(agg)

        print("\n### Confidence interval - BLEU")
        print(agg2)
    else:
        for metric in METRICS:
            agg, agg2 = agg_results[metric.lower()]
            print(f"\n### Summary statistics - {metric}")
            print(agg)

            print(f"\n### Confidence interval - {metric}")
            print(agg2)

    if output_file:
        print("Outputting to CSV...")
        if legacy_mode:
            raw.stack().reset_index().to_csv(output_file, index=False)
        else:
            raw.to_csv(output_file, index=False)



if __name__ == "__main__":
    main()
