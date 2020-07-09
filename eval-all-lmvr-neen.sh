#!/bin/bash

get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# NE - EN
for cp in $(ls ./checkpoints/*lmvr-ne*/checkpoints_ne_en/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    echo 'Seed:'$_seed
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src ne --tgt en \
        --eval-on test \
        --data-folder ./data/wiki_ne_en_lmvr \
        --data-bin-folder ./data-bin/wiki_ne_en_lmvr \
        --model-checkpoint $cp \
        --model-type lmvr \
        --output-file ./translation-output/lmvr/seed-${_seed}/ne-en.output.raw \
        --reference ./data/wiki_ne_en_lmvr/test.en \
        --cuda-device 0
done
