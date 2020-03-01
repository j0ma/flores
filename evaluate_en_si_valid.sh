CHECKPOINT_DIR="/checkpoints/flores/checkpoints_en_si/checkpoint_best.pt"

fairseq-generate \
    data-bin/wiki_si_en_bpe5000/ \
    --source-lang en --target-lang si \
    --path $CHECKPOINT_DIR \
    --beam 5 --lenpen 1.2 \
    --gen-subset valid \
    --remove-bpe=sentencepiece \
    --sacrebleu
