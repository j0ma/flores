# ad hoc script to evaluate nonjoint seed exp

EVAL_SCRIPT="./evaluate_nonjoint.sh"
BPE_SIZE=2500
CUDA_DEVICE=0

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
    echo $LANG
    RESULTS_DIR="./evaluate/${f}"
    CHECKPOINT_DIR="./checkpoints/${f}"
    $EVAL_SCRIPT "en" $LANG $BPE_SIZE $CUDA_DEVICE $RESULTS_DIR $CHECKPOINT_DIR
    $EVAL_SCRIPT $LANG "en" $BPE_SIZE $CUDA_DEVICE $RESULTS_DIR $CHECKPOINT_DIR
done


