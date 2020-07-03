import click
import sys
import os
import re

def read_lines(fp):
    with open(fp, 'r') as f:
       lines = [l.strip() for l in f.readlines()] 

       return lines

def read_lmvr_segmentations(fp):
    lines = read_lines(fp)
    out = []
    curr_sent = []

    for line in lines:
        if line == '':
            out.append(" ".join(curr_sent))
            curr_sent = []
        else:
            # otherwise just add
            curr_sent.append(line)

    return out


@click.command()
@click.option('--input-path')
@click.option('--output-path')
def main(input_path, output_path):
    sentences = read_lmvr_segmentations(input_path)
    with open(output_path, 'w') as f:
        f.writelines(s+"\n" for s in sentences)

if __name__ == "__main__":
    main()
