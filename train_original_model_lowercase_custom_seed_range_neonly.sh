# This script runs the seed experiments using a custom BPE size
# Meant to be run from inside Makefile, can be run in master branch
# provided that the different size BPE data sets exist.

# NOTE: as of 6/18/2020 this only trains/evaluates
#       settings where EN is the target language

BASE_SLUG=$1
BPE_SIZE=$2

FROM_SEED=$3
if [ -z $FROM_SEED ]
then
    FROM_SEED=10
fi

TO_SEED=$4
if [ -z $TO_SEED ]
then
    TO_SEED=19
fi

CUDA_DEVICE=$5

TRAIN_SCRIPT="./train_original_model_lowercase_customseed.sh "
EVAL_SCRIPT="./evaluate_lowercase.sh"

for SEED in $(seq $FROM_SEED $TO_SEED);
do

    SLUG=$BASE_SLUG"-seed"$SEED"-lowercase"

    # 0. create log & checkpoint folder
    LOG_FOLDER=$(bash ./create_log_folder.sh $SLUG)
    CHECKPOINT_FOLDER=$(bash ./create_checkpoint_folder.sh $SLUG)
    echo $LOG_FOLDER
    echo $CHECKPOINT_FOLDER

    # 1. Train NE - EN
    bash $TRAIN_SCRIPT "ne" "en" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

    # 2. Train EN - NE
    bash $TRAIN_SCRIPT "en" "ne" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

    # 5. create results folder
    RESULTS_FOLDER=$(bash ./create_results_folder.sh $SLUG)
    echo $RESULTS_FOLDER

    # 6. Evaluate NE - EN
    bash $EVAL_SCRIPT "ne" "en" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER $CHECKPOINT_FOLDER

    # 7. Evaluate EN - NE
    bash $EVAL_SCRIPT "en" "ne" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER $CHECKPOINT_FOLDER

done
