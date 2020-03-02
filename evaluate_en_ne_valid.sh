CHECKPOINT_DIR="/checkpoints/flores/checkpoints_en_ne/checkpoint_best.pt"

fairseq-generate \
    data-bin/wiki_ne_en_bpe5000/ \
    --source-lang en --target-lang ne \
    --path $CHECKPOINT_DIR \
    --beam 5 --lenpen 1.2 \
    --gen-subset valid \
    --remove-bpe=sentencepiece # note: no sacrebleu here
