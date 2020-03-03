echo "================ FLORES BASELINE REPRODUCTION SCRIPT ================"

echo """
About to train the following language pairs:
- NE-EN
- EN-NE
- SI-EN
- EN-SI
"""

TRAIN_SCRIPT_FOLDER="./train"
LOG_FOLDER="./log"
TIME_SUFFIX=$(date -Iminutes | sed s/':'/'-'/g)

# 1. Train NE - EN
SRC_LANG="ne"
TGT_LANG="en"

echo "About to train baseline for NE-EN ..."

# create path to training file
TRAIN_SCRIPT="train_baseline_ne_en.sh"
TRAIN_PATH="$TRAIN_SCRIPT_FOLDER/$TRAIN_SCRIPT"
echo "Training script located at: $TRAIN_PATH"

# create path for log file
LOG_FILE=$(echo $TRAIN_SCRIPT | sed s/".sh"/"-$TIME_SUFFIX.log"/g)
LOG_OUTPUT_PATH="$LOG_FOLDER/$LOG_FILE"
echo "Logging output to: $LOG_OUTPUT_PATH"

# create path to checkpoint directory
if [ -z ${CHECKPOINT_DIR_NEEN+IMEMPTY} ]; 
then 
    echo "Checkpoint directory unset! Setting to default value..."
    export CHECKPOINT_DIR_NEEN="./checkpoints/checkpoints_ne_en"
    echo "CHECKPOINT_DIR_NEEN is set to '$CHECKPOINT_DIR_NEEN'"; 
else 
    echo "CHECKPOINT_DIR_NEEN is set to '$CHECKPOINT_DIR_NEEN'"; 
fi
echo "Creating checkpoint directory if it doesn't exist..."
mkdir -p $CHECKPOINT_DIR_NEEN

# actually run the training script and pass in necessary env variable
echo "Beginning training..."
TRAIN_PREFIX="CHECKPOINT_DIR=\"$CHECKPOINT_DIR_NEEN\"" 
TRAIN_CMD="$TRAIN_PREFIX $TRAIN_PATH"
eval $TRAIN_CMD > $LOG_OUTPUT_PATH

