#!/bin/bash

BPE_SIZES=(1000 2500 7500 10000)
set -euo pipefail

for model in "lmvr" "lmvr-tuned" "subword-nmt" "morsel" "baseline"
do
    for seed in $(seq 10 14)
    do
        mkdir -p "$(pwd)/translation-output/${model}/seed-${seed}"
        mkdir -p "$(pwd)/translation-output-wmt19/${model}/seed-${seed}"
        mkdir -p "$(pwd)/translation-output-wmt19-additional/${model}/seed-${seed}"
        for bpe_size in "${BPE_SIZES[@]}"; do
            mkdir -p "$(pwd)/translation-output-wmt19-bpe${bpe_size}/${model}/seed-${seed}"
        done
    done
done
