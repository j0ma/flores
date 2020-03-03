evaluate_fairseq () {
    SRC_LANG=$1
    TGT_LANG=$2
    CHECKPOINT_DIR=$3
    DATA_DIR=$4

    if [ "$SRC_LANG" = "en" ];
    then
        fairseq-generate \
            $DATA_DIR \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --path $CHECKPOINT_DIR \
            --beam 5 --lenpen 1.2 \
            --gen-subset test \
            --remove-bpe=sentencepiece # note: no sacrebleu here
    else
        fairseq-generate \
            $DATA_DIR \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --path $CHECKPOINT_DIR \
            --beam 5 --lenpen 1.2 \
            --gen-subset test \
            --remove-bpe=sentencepiece \
            --sacrebleu
    fi
}

evaluate () {
    SRC_LANG=$1
    TGT_LANG=$2
    TIME_SUFFIX=$(date -Iminutes | sed s/':'/'-'/g)
    RESULTS_FOLDER="./evaluate/"$TIME_SUFFIX
    mkdir -p $RESULTS_FOLDER

    # create path for log file
    RESULTS_FILE="baseline_"$SRC_LANG"_"$TGT_LANG".log"
    RESULTS_OUTPUT_PATH="$RESULTS_FOLDER/$RESULTS_FILE"
    echo "Saving output to: $RESULTS_OUTPUT_PATH"

    # create path to checkpoint directory
    if [ -z ${CHECKPOINT_DIR+IMEMPTY} ]; 
    then 
        echo "Checkpoint directory unset! Setting to default value..."
        CHECKPOINT_DIR="./checkpoints/checkpoints_"$SRC_LANG"_"$TGT_LANG
        echo "CHECKPOINT_DIR is set to '$CHECKPOINT_DIR'"; 
    else 
        echo "CHECKPOINT_DIR is set to '$CHECKPOINT_DIR'"; 
    fi

    # infer data directory
    if [ "$SRC_LANG" = "si" ] || [ "$TGT_LANG" = "si" ];
    then
        DATA_DIR="data-bin/wiki_si_en_bpe5000/"
    else
        DATA_DIR="data-bin/wiki_ne_en_bpe5000/"
    fi
    echo "Data folder is: "$DATA_DIR

    echo "About to evaluate..."
    evaluate_fairseq $SRC_LANG $TGT_LANG $CHECKPOINT_DIR $DATA_DIR > $RESULTS_OUTPUT_PATH

}

# 1. Evaluate NE - EN
evaluate "ne" "en"

# 2. Evaluate EN - NE
evaluate "en" "ne"

# 3. Evaluate SI - EN
evaluate "si" "en"

# 4. Evaluate EN - SI
evaluate "en" "si"

