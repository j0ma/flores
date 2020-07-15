#!/bin/bash

get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# NE - EN
for cp in $(ls ./checkpoints/*morsel-ne*/checkpoints_ne_en/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    echo 'Seed:'$_seed
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src ne --tgt en \
        --eval-on test \
        --data-folder ./data/wiki_ne_en_morsel \
        --data-bin-folder ./data-bin/wiki_ne_en_morsel \
        --model-checkpoint $cp \
        --model-type morsel \
        --output-file ./translation-output/morsel/seed-${_seed}/ne-en.output.raw \
        --reference ./data/wiki_ne_en_morsel/test.en \
        --remove-bpe "off" &
done
wait
