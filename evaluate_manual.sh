# this script is only for MANUAL evaluation, 
# of a given experiment, ie. the log folder must 
# be provided as $1 along with an optional slug representing
# the experiment name given as $2

evaluate_fairseq () {
    SRC_LANG=$1
    TGT_LANG=$2
    CHECKPOINT_PATH=$3
    DATA_DIR=$4

    if [ "$SRC_LANG" = "en" ];
    then
        fairseq-generate \
            $DATA_DIR \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --path $CHECKPOINT_PATH \
            --beam 5 --lenpen 1.2 \
            --gen-subset test \
            --remove-bpe=sentencepiece # note: no sacrebleu here
    else
        fairseq-generate \
            $DATA_DIR \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --path $CHECKPOINT_PATH \
            --beam 5 --lenpen 1.2 \
            --gen-subset test \
            --remove-bpe=sentencepiece \
            --sacrebleu
    fi
}

evaluate () {

    SRC_LANG=$1
    TGT_LANG=$2
    LOG_FOLDER=$3
    BPE_SIZE=$4
    RESULTS_DIR="./evaluate/"$(ls -t ./evaluate | head -1)

    echo "BPE size is: "$BPE_SIZE

    # create path for log file
    RESULTS_FILE="baseline_"$SRC_LANG"_"$TGT_LANG".log"
    RESULTS_OUTPUT_PATH="$RESULTS_DIR/$RESULTS_FILE"
    echo "Saving output to: $RESULTS_OUTPUT_PATH"

    # infer checkpoint directory
    #TIMESTAMP=$(ls -t ./checkpoints/ | head -n 1)
    #CHECKPOINT_DIR="./checkpoints/"$TIMESTAMP"/checkpoints_"$SRC_LANG"_"$TGT_LANG

    CHECKPOINT_DIR=$(head $LOG_FOLDER"/baseline_"$SRC_LANG"_"$TGT_LANG".log" | \
                     head -4 | \
                     tail -1 | \
                     grep -o './checkpoints.*$' | \
                     sed s/\'//g)

    CHECKPOINT_PATH=$CHECKPOINT_DIR"/checkpoint_best.pt"
    echo "CHECKPOINT_PATH is: $CHECKPOINT_PATH";

    # infer data directory
    if [ "$SRC_LANG" = "si" ] || [ "$TGT_LANG" = "si" ];
    then
        DATA_DIR="data-bin/wiki_si_en_bpe"$BPE_SIZE"/"
    else
        DATA_DIR="data-bin/wiki_ne_en_bpe"$BPE_SIZE"/"
    fi
    echo "Data folder is: "$DATA_DIR
    echo "About to evaluate..."
    evaluate_fairseq $SRC_LANG $TGT_LANG $CHECKPOINT_PATH $DATA_DIR > $RESULTS_OUTPUT_PATH

}

LOG_PATH=$1
SLUG=$2
BPE_SIZE=$3

if [ -z $BPE_SIZE ]
then
    BPE_SIZE=5000
fi

echo "Creating results folder..."
bash ./create_results_folder.sh $SLUG

evaluate "ne" "en" $LOG_PATH $BPE_SIZE
evaluate "en" "ne" $LOG_PATH $BPE_SIZE
evaluate "si" "en" $LOG_PATH $BPE_SIZE
evaluate "en" "si" $LOG_PATH $BPE_SIZE
