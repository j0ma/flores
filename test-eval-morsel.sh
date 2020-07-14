#!/bin/bash

src=$1
tgt=$2
cp=$3
_seed=$4

bash eval-model.sh \
    --src "${src}" --tgt "${tgt}" \
    --eval-on test \
    --data-folder ./data/wiki_ne_en_morsel \
    --data-bin-folder ./data-bin/wiki_ne_en_morsel \
    --model-checkpoint "${cp}" \
    --model-type morsel \
    --output-file ./translation-output/morsel/seed-${_seed}/${src}-${tgt}.output.raw \
    --reference ./data/wiki_ne_en_morsel/test.$tgt
