import itertools as it
import sys
import os
import re

def flatten(nested):
    return list(it.chain.from_iterable(nested))

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

def compute_lines(path, lang):
    data = {}
    files = ['train.{}'.format(lang), 
            'train.{}.tok'.format(lang), 
            'train.{}.tok.lower'.format(lang), 
            'train.lmvr.intermediate.{}'.format(lang), 
            'train.lmvr.{}'.format(lang)]
    for f in files:
        if 'intermediate' in f:
            data[f] = read_lmvr_segmentations("{}/{}".format(path, f))
        else:
            data[f] = read_lines("{}/{}".format(path, f))

    output_lines = [
        'Sentences in "{}": {}'.format(f, len(data[f]))
        for f in files
    ]
    return '\n'.join(output_lines)

if __name__ == '__main__':
    path = sys.argv[1]
    lang = sys.argv[2]
    report = compute_lines(path, lang)
    print(report)



