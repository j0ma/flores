get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# EN - NE
for cp in $(ls ./checkpoints/*morsel-ne*/checkpoints_en_ne/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src en --tgt ne \
        --eval-on test \
        --data-folder ./data/wiki_ne_en_morsel \
        --data-bin-folder ./data-bin/wiki_ne_en_morsel \
        --model-checkpoint $cp \
        --model-type morsel \
        --output-file ./translation-output/morsel/seed-${_seed}/en-ne.output.raw \
        --reference ./data/wiki_ne_en_morsel/test.ne \
        --remove-bpe "" &
done
wait
