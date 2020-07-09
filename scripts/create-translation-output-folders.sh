#!/bin/bash

set -euo pipefail

for model in "lmvr" "lmvr-tuned" "subword-nmt" "morsel" "baseline"
do
    for seed in $(seq 10 14)
    do
        mkdir -p "$(pwd)/translation-output/${model}/seed-${seed}"
    done
done
