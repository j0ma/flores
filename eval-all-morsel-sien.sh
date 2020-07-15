get_seed() {
    echo $1 | \
        grep -Eo "seed[0-9]{2}" | \
        sed "s/seed//g"
}

# SI - EN
for cp in $(ls ./checkpoints/*morsel-si*/checkpoints_si_en/checkpoint_best.pt); do
    _seed=$(get_seed $cp)
    [ -z $_seed ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src si --tgt en \
        --eval-on test \
        --data-folder ./data/wiki_si_en_morsel \
        --data-bin-folder ./data-bin/wiki_si_en_morsel \
        --model-checkpoint $cp \
        --model-type morsel \
        --output-file ./translation-output/morsel/seed-${_seed}/si-en.output.raw \
        --reference ./data/wiki_si_en_morsel/test.en \
        --remove-bpe "off" &
done
wait
