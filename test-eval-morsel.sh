#!/bin/bash

src=$1
tgt=$2
cp=$3
_seed=$4

if [ "${src}" = "si" ] || [ "${tgt}" = "si" ]; then
    foreign="si"
else
    foreign="ne"
fi

bash eval-model.sh \
    --src "${src}" --tgt "${tgt}" \
    --eval-on test \
    --data-folder "./data/wiki_${foreign}_en_morsel" \
    --data-bin-folder "./data-bin/wiki_${foreign}_en_morsel" \
    --model-checkpoint "${cp}" \
    --model-type morsel \
    --output-file ./translation-output/morsel/seed-${_seed}/${src}-${tgt}.output.raw \
    --reference "./data/wiki_${foreign}_en_morsel/test.${tgt}" \
    --remove-bpe "off"
