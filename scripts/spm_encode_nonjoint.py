# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import contextlib
import sys

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_src", required=True,
                        help="sentencepiece model to use for encodingi src")
    parser.add_argument("--model_tgt", required=True,
                        help="sentencepiece model to use for encoding tgt")
    parser.add_argument("--inputs", nargs="+", default=['-'],
                        help="input files to filter/encode")
    parser.add_argument("--outputs", nargs="+", default=['-'],
                        help="path to save encoded outputs")
    parser.add_argument("--output_format", choices=["piece", "id"], default="piece")
    parser.add_argument("--min-len", type=int, metavar="N",
                        help="filter sentence pairs with fewer than N tokens")
    parser.add_argument("--max-len", type=int, metavar="N",
                        help="filter sentence pairs with more than N tokens")
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
            "number of input and output paths should match"

    sp_src = spm.SentencePieceProcessor()
    sp_src.Load(args.model_src)
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.Load(args.model_tgt)

    if args.output_format == "piece":
        def encode(l, sp=sp_src):
            return sp.EncodeAsPieces(l)
    elif args.output_format == "id":
        def encode(l, sp=sp_src):
            #return list(map(str, sp.EncodeAsIds(l)))
            return [str(x) for x in sp.EncodeAsIds(l)]
    else:
        raise NotImplementedError

    if args.min_len is not None or args.max_len is not None:
        def valid(line):
            return (
                (args.min_len is None or len(line) >= args.min_len)
                and (args.max_len is None or len(line) <= args.max_len)
            )
    else:
        def valid(lines):
            return True

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8", newline="\n", errors="ignore"))
                if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8", newline="\n"))
                if output != "-" else sys.stdout
            for output in args.outputs
        ]

        stats = {
            "num_empty": 0,
            "num_filtered": 0,
        }

        def encode_line(line, sp=sp_src):
            """Convert string into BPE representation"""
            line = line.strip()                 # line -> tokens
            if len(line) > 0:                   # only encode if we found tokens
                line = encode(line, sp=sp_src)  # use SentencePieceProcessor to encode
                if valid(line):                 # validate the encoded pieces
                    return line                 #   -> return if valid
                else:                           # 
                    stats["num_filtered"] += 1  #   -> otherwise take note & return None
            else:                               # in case of empty line:
                stats["num_empty"] += 1         #   -> record that we had an empty line
            return None                         # return None if things fail

        # inputs: tuple of iterables, e.g. src iterable & tgt iterable
        # lines: [(src_line, [optionally_target_line])]
        for i, (src_line, tgt_line) in enumerate(zip(*inputs), start=1):

            # this is a tuple
            #print(lines)

            # invariant: len(lines) == len(enc_lines)
            #enc_lines = [encode_line(line) for line in lines]
            src_enc = encode_line(src_line, sp=sp_src)
            tgt_enc = encode_line(tgt_line, sp=sp_tgt)
            src_not_none = src_enc is not None
            tgt_not_none = tgt_enc is not None

            # only output if there were no None
            # if not any(enc_line is None for enc_line in enc_lines):
                # for enc_line, output_h in zip(enc_lines, outputs):
                    # print(" ".join(enc_line), file=output_h)

            if src_not_none and tgt_not_none:
                src_output, tgt_output = outputs
                print(" ".join(src_enc), file=src_output)
                print(" ".join(tgt_enc), file=tgt_output)
                    
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        print("skipped {} empty lines".format(stats["num_empty"]), file=sys.stderr)
        print("filtered {} lines".format(stats["num_filtered"]), file=sys.stderr)


if __name__ == "__main__":
    main()
