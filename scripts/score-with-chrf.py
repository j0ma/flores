from nltk.translate.chrf_score import corpus_chrf
import click
import sys


def read_lines(fp):
    with open(fp, "r") as f:
        lines = [line.lower().strip() for line in f.readlines()]

        return lines


@click.command()
@click.option("--hypotheses-file")
@click.option("--references-file")
@click.option("--output-file")
@click.option("--min-len", default=1)
@click.option("--max-len", default=6)
@click.option("--beta", default=3)
@click.option("--include-whitespace", is_flag=True, default=False)
def main(
    hypotheses_file,
    references_file,
    output_file,
    min_len,
    max_len,
    beta,
    include_whitespace,
):
    hypotheses = [h.split() for h in read_lines(hypotheses_file)]
    references = [r.split() for r in read_lines(references_file)]
    ignore_whitespace = not include_whitespace
    score = corpus_chrf(
        references=references,
        hypotheses=hypotheses,
        min_len=min_len,
        max_len=max_len,
        beta=beta,
        ignore_whitespace=ignore_whitespace,
    )
    score = round(score, 3)
    

    output_msg=f"ChrF{beta} = {score}\n"
    if output_file == "-":
        sys.stdout.write(output_msg)
    else:
        with open(output_file, 'w') as f:
            f.write(output_msg)


if __name__ == "__main__":
    main()
