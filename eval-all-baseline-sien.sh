#!/bin/bash

get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# NE - EN
for cp in $(ls ./checkpoints/*exp333*lowercase*/checkpoints_si_en/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    echo 'Seed:'$_seed
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src si --tgt en \
        --eval-on test \
        --data-folder ./data/wiki_si_en_bpe5000_lowercase \
        --data-bin-folder ./data-bin/wiki_si_en_bpe5000_lowercase \
        --model-checkpoint $cp \
        --model-type bpe \
        --output-file ./translation-output/baseline/seed-${_seed}/si-en.output.raw \
        --reference ./data/wiki_si_en_bpe5000_lowercase/test.en \
        --remove-bpe "sentencepiece" &
done
wait
