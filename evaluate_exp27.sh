# ad hoc script to evaluate nonjoint seed exp

EVAL_SCRIPT="./evaluate_nonjoint.sh"
BPE_SIZE=5000
CUDA_DEVICE=1

detect_lang () {
    si_found=$(echo $1 | grep "\-si\-")
    if [ -z $si_found ];
    then
        echo "ne"
    else
        echo "si"
    fi
}

for f in $(cat exp27-common-folders)
do
    LANG=$(detect_lang $f)
    #echo $LANG
    RESULTS_DIR="./evaluate/${f}"
    CHECKPOINT_DIR="./checkpoints/${f}"
    #echo $RESULTS_DIR
    #echo $CHECKPOINT_DIR
    $EVAL_SCRIPT "en" $LANG $BPE_SIZE $CUDA_DEVICE $RESULTS_DIR "./checkpoints/${f}/checkpoints_en_${LANG}"
    $EVAL_SCRIPT $LANG "en" $BPE_SIZE $CUDA_DEVICE $RESULTS_DIR "./checkpoints/${f}/checkpoints_${LANG}_en"
done


