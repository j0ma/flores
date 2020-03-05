echo "================ FLORES BASELINE REPRODUCTION SCRIPT ================"

SRC_LANG=$1
TGT_LANG=$2
SRC_LANG_CAP=$(echo $SRC_LANG | awk '{print toupper($0)}')
TGT_LANG_CAP=$(echo $TGT_LANG | awk '{print toupper($0)}')

echo "About to train the supervised for the following language pair: "$SRC_LANG_CAP"-"$TGT_LANG_CAP

train_fairseq () {

    SRC_LANG=$1
    TGT_LANG=$2
    CHECKPOINT_DIR=$3

    CUDA_VISIBLE_DEVICES=0 fairseq-train \
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
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
        --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
        --lr 1e-3 --min-lr 1e-9 \
        --max-tokens 4000 \
        --update-freq 4 \
        --max-epoch 100 \
        --save-interval 10 \
        --save-dir $CHECKPOINT_DIR
}

train () {

    SRC_LANG=$1
    TGT_LANG=$2
    SRC_LANG_CAP=$(echo $SRC_LANG | awk '{print toupper($0)}')
    TGT_LANG_CAP=$(echo $TGT_LANG | awk '{print toupper($0)}')
    echo "About to train baseline for $SRC_LANG_CAP - $TGT_LANG_CAP ..."

    TIME_SUFFIX=$(date -Iminutes | sed s/':'/'-'/g)
    LOG_DIR="./log/"$TIME_SUFFIX
    mkdir -p $LOG_DIR

    # create path for log file
    LOG_FILE="baseline_"$SRC_LANG"_"$TGT_LANG".log"
    LOG_OUTPUT_PATH="$LOG_DIR/$LOG_FILE"
    echo "Logging output to: $LOG_OUTPUT_PATH"

    # create path to checkpoint directory
    CHECKPOINT_DIR="./checkpoints/checkpoints_"$SRC_LANG"_"$TGT_LANG
    echo "Checkpoint directory unset! Setting to default value..."
    echo "CHECKPOINT_DIR is set to '$CHECKPOINT_DIR'"; 
    echo "Creating checkpoint directory if it doesn't exist..."
    mkdir -p $CHECKPOINT_DIR

    # infer data directory
    if [ "$SRC_LANG" = "si" ] || [ "$TGT_LANG" = "si" ];
    then
        DATA_DIR="data-bin/wiki_si_en_bpe5000/"
    else
        DATA_DIR="data-bin/wiki_ne_en_bpe5000/"
    fi
    
    echo "Data folder is: "$DATA_DIR

    # actually run the training script and pass in necessary env variable
    echo "Beginning training..."
    train_fairseq $SRC_LANG $TGT_LANG $CHECKPOINT_DIR $DATA_DIR > $LOG_OUTPUT_PATH
}

train $1 $2

