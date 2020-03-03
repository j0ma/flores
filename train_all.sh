echo "================ FLORES BASELINE REPRODUCTION SCRIPT ================"

echo """
About to train the following language pairs:
- NE-EN
- EN-NE
- SI-EN
- EN-SI
"""
train () {

    SRC_LANG=$1
    TGT_LANG=$2
    SRC_LANG_CAP=$(echo $SRC_LANG | awk '{print toupper($0)}')
    TGT_LANG_CAP=$(echo $TGT_LANG | awk '{print toupper($0)}')
    echo "About to train baseline for $SRC_LANG_CAP - $TGT_LANG_CAP ..."

    TRAIN_SCRIPT_FOLDER="./train"
    LOG_FOLDER="./log"
    TIME_SUFFIX=$(date -Iminutes | sed s/':'/'-'/g)


    # create path to training file
    TRAIN_SCRIPT="train_baseline_"$SRC_LANG"_"$TGT_LANG".sh"
    TRAIN_PATH="$TRAIN_SCRIPT_FOLDER/$TRAIN_SCRIPT"
    echo "Training script located at: $TRAIN_PATH"

    # create path for log file
    LOG_FILE=$(echo $TRAIN_SCRIPT | sed s/".sh"/"-$TIME_SUFFIX.log"/g)
    LOG_OUTPUT_PATH="$LOG_FOLDER/$LOG_FILE"
    echo "Logging output to: $LOG_OUTPUT_PATH"

    # create path to checkpoint directory
    if [ -z ${CHECKPOINT_DIR+IMEMPTY} ]; 
    then 
        echo "Checkpoint directory unset! Setting to default value..."
        CHECKPOINT_DIR="./checkpoints/checkpoints_"$SRC_LANG"_"$TGT_LANG
        echo "CHECKPOINT_DIR is set to '$CHECKPOINT_DIR'"; 
    else 
        echo "CHECKPOINT_DIR is set to '$CHECKPOINT_DIR'"; 
    fi
    echo "Creating checkpoint directory if it doesn't exist..."

    # actually run the training script and pass in necessary env variable
    echo "Beginning training..."
    TRAIN_PREFIX="CHECKPOINT_DIR=\"$CHECKPOINT_DIR\"" 
    TRAIN_CMD="$TRAIN_PREFIX $TRAIN_PATH"
    eval $TRAIN_CMD #> $LOG_OUTPUT_PATH
    
    echo
}

# 1. Train NE - EN
train "ne" "en"

# 2. Train EN - NE
train "en" "ne"

# 3. Train SI - EN
train "si" "en"

# 4. Train EN - SI
train "en" "si"

