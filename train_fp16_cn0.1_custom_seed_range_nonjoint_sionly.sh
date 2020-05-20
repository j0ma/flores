# This script runs the seed experiments using a custom BPE size
# Meant to be run from inside Makefile, can be run in master branch
# provided that the different size BPE data sets exist.

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

TRAIN_SCRIPT="./train_fp16_cn0.1_customseed_nonjoint.sh "
EVAL_SCRIPT="./evaluate_nonjoint.sh"

for SEED in $(seq $FROM_SEED $TO_SEED);
do

    SLUG=$BASE_SLUG"-seed"$SEED"-nonjoint"

    # 0. create log & checkpoint folder
    LOG_FOLDER=$(bash ./create_log_folder.sh $SLUG)
    CHECKPOINT_FOLDER=$(bash ./create_checkpoint_folder.sh $SLUG)
    echo $LOG_FOLDER
    echo $CHECKPOINT_FOLDER

    # 3. Train SI - EN
    bash $TRAIN_SCRIPT "si" "en" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

    # 4. Train EN - SI
    bash $TRAIN_SCRIPT "en" "si" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

    # 5. create results folder
    RESULTS_FOLDER=$(bash ./create_results_folder.sh $SLUG)
    echo $RESULTS_FOLDER

    # 8. Evaluate SI - EN
    bash $EVAL_SCRIPT "si" "en" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER

    # 9. Evaluate EN - SI
    bash $EVAL_SCRIPT "en" "si" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER

done
