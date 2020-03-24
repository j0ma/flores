# harameters.sh
#
# description
# -----------
# given the path to an evaluation output folder,
# finds the parameter configuration used in the experiment.
#
# notes
# -----
# can be outputted to text editor of choice very nicely.

EVAL_FOLDER=$1
SRC_LANG='ne' # these do not need to be passed in as args since each
TGT_LANG='en' # experiment is run with same params for all languages

# find the eval file
EVAL_FILE=$EVAL_FOLDER"/baseline_"$SRC_LANG"_"$TGT_LANG".log"

# given eval file, find checkpoint folder & timestamp
# | loading model(s) from ./checkpoints/2020-03-21T16-02-04-00/checkpoints_ne_en/checkpoint_best.pt
CHECKPOINT_FILE=$(grep "loading model(s) from" $EVAL_FILE |
                 sed "s/^| loading model(s) from //g")
echo $CHECKPOINT_FILE

TIMESTAMP=$(echo $CHECKPOINT_FILE | grep -o "2020.*00")

# given time stamp, find log folder
LOG_FOLDER="./log/"$TIMESTAMP""
LOG_FILE=$LOG_FOLDER"/baseline_"$SRC_LANG"_"$TGT_LANG".log"

# get the Namespace information from log file and format nicely

# get the typical parameters of interest, i.e.
# batch size, fp16, learning rate, min_lr, clip_norm, seed
echo "Eval file: "$EVAL_FILE
echo "Log file: "$LOG_FILE
echo "Hyperparameters: "
head $LOG_FILE | \
    tail -2 | \
    head -1 | \
    sed "s/, /\n/g" | \
    sed "s/Namespace(/Namespace(\n/g" |
    grep "max_tokens=\|fp16=\|lr=\[\|min_lr=\|clip_norm=\|seed"
