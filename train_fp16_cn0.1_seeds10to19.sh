# This script runs the seed experiments using a custom BPE size
# Meant to be run from inside Makefile, can be run in master branch
# provided that the different size BPE data sets exist.

BASE_SLUG=$1
BPE_SIZE=$2

for SEED in $(seq 10 19);
do

    SLUG=$BASE_SLUG"-seed"$SEED

    # 0. create log & checkpoint folder
    bash ./create_log_folder.sh $SLUG
    bash ./create_checkpoint_folder.sh $SLUG

    # 1. Train NE - EN
    bash ./train_fp16_cn0.1_customseed.sh "ne" "en" $SEED $BPE_SIZE

    # 2. Train EN - NE
    bash ./train_fp16_cn0.1_customseed.sh "en" "ne" $SEED $BPE_SIZE

    # 3. Train SI - EN
    bash ./train_fp16_cn0.1_customseed.sh "si" "en" $SEED $BPE_SIZE

    # 4. Train EN - SI
    bash ./train_fp16_cn0.1_customseed.sh "en" "si" $SEED $BPE_SIZE

    # 5. create results folder
    bash ./create_results_folder.sh $SLUG

    # 6. Evaluate NE - EN
    bash ./evaluate.sh "ne" "en"

    # 7. Evaluate EN - NE
    bash ./evaluate.sh "en" "ne"

    # 8. Evaluate SI - EN
    bash ./evaluate.sh "si" "en"

    # 9. Evaluate EN - SI
    bash ./evaluate.sh "en" "si"

done
