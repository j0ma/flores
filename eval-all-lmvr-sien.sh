get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# SI - EN
for cp in $(ls ./checkpoints/*lmvr-si*/checkpoints_si_en/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src si --tgt en \
        --eval-on test \
        --data-folder ./data/wiki_si_en_lmvr \
        --data-bin-folder ./data-bin/wiki_si_en_lmvr \
        --model-checkpoint $cp \
        --model-type lmvr \
        --output-file ./translation-output/lmvr/seed-${_seed}/si-en.output.raw \
        --reference ./data/wiki_si_en_lmvr/test.en \
        --remove-bpe "regular"
done
