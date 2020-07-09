get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# EN - NE
for cp in $(ls ./checkpoints/*lmvr-ne*/checkpoints_en_ne/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src en --tgt ne \
        --eval-on test \
        --data-folder ./data/wiki_ne_en_lmvr \
        --data-bin-folder ./data-bin/wiki_ne_en_lmvr \
        --model-checkpoint $cp \
        --model-type lmvr \
        --output-file ./translation-output/lmvr/seed-${_seed}/en-ne.output.raw \
        --reference ./data/wiki_ne_en_lmvr/test.ne \
        --cuda-device 0
done

