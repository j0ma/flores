train_fairseq() {

    SRC_LANG=$1
    TGT_LANG=$2
    CHECKPOINT_DIR=$3
    DATA_DIR=$4
    RAND_SEED=$5
    CUDA_DEVICE=$6

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE fairseq-train \
        $DATA_DIR \
        --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --arch transformer --share-all-embeddings \
        --encoder-layers 5 --decoder-layers 5 \
        --encoder-embed-dim 512 --decoder-embed-dim 512 \
        --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
        --encoder-attention-heads 2 --decoder-attention-heads 2 \
        --encoder-normalize-before --decoder-normalize-before \
        --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
        --weight-decay 0.0001 \
        --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
        --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
        --lr 1e-3 --min-lr 1e-9 \
        --max-tokens 4000 \
        --update-freq 4 \
        --max-epoch 100 \
        --save-interval 10 \
        --save-dir $CHECKPOINT_DIR \
        --seed $RAND_SEED \
        --fp16
}

train() {

    SRC_LANG=$1
    TGT_LANG=$2
    RAND_SEED=$3
    #LOG_DIR="./log/"$(ls -t ./log | head -1)
    SRC_LANG_CAP=$(echo $SRC_LANG | awk '{print toupper($0)}')
    TGT_LANG_CAP=$(echo $TGT_LANG | awk '{print toupper($0)}')
    BPE_SIZE=$4
    CUDA_DEVICE=$5
    LOG_DIR=$6
    CHECKPOINT_DIR=$7
    CHECKPOINT_DIR="${CHECKPOINT_DIR}/checkpoints_${SRC_LANG}_${TGT_LANG}"

    if [ -z $BPE_SIZE ]; then
        BPE_SIZE=5000
    fi

    # create path for log file
    LOG_FILE="baseline_"$SRC_LANG"_"$TGT_LANG".log"
    LOG_OUTPUT_PATH="$LOG_DIR/$LOG_FILE"

    echo "================ FLORES BASELINE WITH CLIP_NORM=0.1, BPE=$BPE_SIZE AND SEED=$RAND_SEED USING SUBWORD-NMT ================" >>$LOG_OUTPUT_PATH
    echo "About to train the supervised for the following language pair: "$SRC_LANG_CAP"-"$TGT_LANG_CAP >>$LOG_OUTPUT_PATH
    echo "Logging output to: $LOG_OUTPUT_PATH"

    # create path to checkpoint directory
    echo "CHECKPOINT_DIR is set to '$CHECKPOINT_DIR'" >>$LOG_OUTPUT_PATH
    echo "Creating checkpoint directory if it doesn't exist..." >>$LOG_OUTPUT_PATH
    mkdir -p $CHECKPOINT_DIR

    # infer data directory
    if [ "$SRC_LANG" = "si" ] || [ "$TGT_LANG" = "si" ]; then
        DATA_DIR="data-bin/wiki_si_en_bpe${BPE_SIZE}_subwordnmt/"
    else
        DATA_DIR="data-bin/wiki_ne_en_bpe${BPE_SIZE}_subwordnmt/"
    fi

    echo "Data folder is: "$DATA_DIR >>$LOG_OUTPUT_PATH

    # actually run the training script and pass in necessary env variable
    echo "Beginning training..." >>$LOG_OUTPUT_PATH
    echo "Time at beginning: "$(date) >>$LOG_OUTPUT_PATH
    train_fairseq $SRC_LANG $TGT_LANG $CHECKPOINT_DIR $DATA_DIR $RAND_SEED $CUDA_DEVICE >>$LOG_OUTPUT_PATH
    echo "Done training." >>$LOG_OUTPUT_PATH
    echo "Time at end: "$(date) >>$LOG_OUTPUT_PATH
}

train $1 $2 $3 $4 $5 $6 $7
