SLUG=$1
export TIME_SUFFIX=$(date -Iminutes | sed s/":"/"-"/g)
if [ -z $SLUG ]
then
    export CHECKPOINT_FOLDER="./checkpoints/"$TIME_SUFFIX
else
    export CHECKPOINT_FOLDER="./checkpoints/"$TIME_SUFFIX"-"$SLUG
fi

mkdir -p $CHECKPOINT_FOLDER
echo $CHECKPOINT_FOLDER
