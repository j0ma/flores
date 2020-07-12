get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# EN - SI
for cp in $(ls ./checkpoints/*lmvr-si*/checkpoints_en_si/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src en --tgt si \
        --eval-on test \
        --data-folder ./data/wiki_si_en_lmvr \
        --data-bin-folder ./data-bin/wiki_si_en_lmvr \
        --model-checkpoint $cp \
        --model-type lmvr \
        --output-file ./translation-output/lmvr/seed-${_seed}/en-si.output.raw \
        --reference ./data/wiki_si_en_lmvr/test.si \
        --remove-bpe "regular"
done
