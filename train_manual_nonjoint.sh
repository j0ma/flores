# For loop from train_fp16_cn0.1_custom_seed_range.sh

BASE_SLUG=$1
BPE_SIZE=$2
SEED=$3
CUDA_DEVICE=$4
SLUG=$BASE_SLUG"-seed"$SEED"-nonjoint"

TRAIN_SCRIPT="./train_fp16_cn0.1_customseed_nonjoint.sh "
EVAL_SCRIPT="./evaluate_nonjoint.sh"

# 0. create log & checkpoint folder
LOG_FOLDER=$(bash ./create_log_folder.sh $SLUG)
CHECKPOINT_FOLDER=$(bash ./create_checkpoint_folder.sh $SLUG)
echo $LOG_FOLDER
echo $CHECKPOINT_FOLDER

# 4. Train EN - SI
bash $TRAIN_SCRIPT "en" "si" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

# 3. Train SI - EN
bash $TRAIN_SCRIPT "si" "en" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

# 2. Train EN - NE
bash $TRAIN_SCRIPT "en" "ne" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

# 1. Train NE - EN
bash $TRAIN_SCRIPT "ne" "en" $SEED $BPE_SIZE $CUDA_DEVICE $LOG_FOLDER $CHECKPOINT_FOLDER

# 5. create results folder
RESULTS_FOLDER=$(bash ./create_results_folder.sh $SLUG)
echo $RESULTS_FOLDER

# 6. Evaluate NE - EN
bash $EVAL_SCRIPT "ne" "en" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER

# 7. Evaluate EN - NE
bash $EVAL_SCRIPT "en" "ne" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER

# 8. Evaluate SI - EN
bash $EVAL_SCRIPT "si" "en" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER

# 9. Evaluate EN - SI
bash $EVAL_SCRIPT "en" "si" $BPE_SIZE $CUDA_DEVICE $RESULTS_FOLDER

