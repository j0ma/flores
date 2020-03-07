export TIME_SUFFIX=$(date -Iminutes | sed s/":"/"-"/g)
export CHECKPOINT_FOLDER="./checkpoints/"$TIME_SUFFIX
mkdir -p $CHECKPOINT_FOLDER
echo $CHECKPOINT_FOLDER
