import click
import sys
import os
import re

# sys.stdin.reconfigure(encoding="utf-8")
# sys.stdout.reconfigure(encoding="utf-8")
# sys.stderr.reconfigure(encoding="utf-8")

SUPPORTED_MODELS = {"lmvr", "morsel", "lmvr-tuned"}
DOUBLE_PLUS = "⧺"
LMVR_SEP = "+"


def read_lines(fp):
    with open(fp, "r") as f:
        lines = [line.strip() for line in f.readlines()]

        return lines


def read_lmvr_segmentations(fp, sep="@@"):
    lines = read_lines(fp)
    sentences = []
    curr_sent = []

    # stitch together words that belong to the same
    # sentence. result is a list "sentences" that
    # contains one sentence per line

    for line in lines:
        if line == "":
            sentences.append(" ".join(curr_sent))
            curr_sent = []
        else:
            # otherwise just add
            curr_sent.append(line)

    final = " ".join(curr_sent)
    try:
        if final != sentences[-1]:
            sentences.append(final)
    except IndexError:
        sentences.append(final)

    return sentences


def convert_lmvr_to_bpe_notation(sentences, sep="@@"):
    def is_lmvr_suffix(s):
        return s.startswith(LMVR_SEP) and len(s) > 1

    sentences_bpe = []

    # converts from the lmvr format to bpe / subword-nmt
    # format. for instance: amicab +ly ===> amicab@@ ly

    for sent in sentences:
        tokens_lmvr = sent.split(" ")
        tokens_bpe = []

        cur_nxt_pairs = zip(tokens_lmvr, tokens_lmvr[1:])

        for cur, nxt in cur_nxt_pairs:

            if is_lmvr_suffix(cur):
                cur = cur[1:]

            if is_lmvr_suffix(nxt):
                cur = f"{cur}{sep}"
            tokens_bpe.append(cur)

        # now we handle the final token
        final = tokens_lmvr[-1]

        if is_lmvr_suffix(final):
            final = final[1:]
        tokens_bpe.append(final)

        sent_bpe = " ".join(tokens_bpe)
        sentences_bpe.append(sent_bpe)

    return sentences_bpe


def read_morsel_segmentations(fp):
    """
    Read in a corpus of sentences, and un-segment
    words according to the MORSEL notatoin.
    """

    lines = [
        line.replace("@@ ", "")
        .replace(DOUBLE_PLUS + " ", "")
        .replace(" " + DOUBLE_PLUS, "")

        for line in read_lines(fp)
    ]

    return lines


"""
for line_num, line in enumerate(sys.stdin, 1):
    fields = line.rstrip("\n").split(" ")
    if len(fields) != 2:
        raise ValueError(f"Line {line_num} is not properly formatted: {repr(line)}")
    count, word = fields
    if not word.strip():
        print(f"Skipping whitespace word of length {len(word)} on line {line_num}: {repr(line)}", file=sys.stderr)
        continue
"""


@click.command()
@click.option("--input-path")
@click.option("--output-path")
@click.option("--model-type", required=True)
@click.option("--bpe-separator", default="@@")
@click.option("--convert-to-bpe", is_flag=True, default=False)
def main(
    input_path, output_path, model_type, bpe_separator, convert_to_bpe=False
):
    assert model_type in SUPPORTED_MODELS, "Error: Unsupported model!"
    if 'lmvr' in model_type:
        sentences = read_lmvr_segmentations(input_path)
        if convert_to_bpe:
            sentences = convert_lmvr_to_bpe_notation(sentences, bpe_separator)
    elif model_type == 'morsel':
        sentences = read_morsel_segmentations(input_path)
    with open(output_path, "w") as f:
        # f.writelines(s+"\n" for s in sentences)
        f.write("\n".join(sentences)+"\n")

if __name__ == "__main__":
    main()
