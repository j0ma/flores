SLUG=$1
export TIME_SUFFIX=$(date -Iminutes | sed s/":"/"-"/g)
if [ -z $SLUG ]
then
    export LOG_FOLDER="./log/"$TIME_SUFFIX
else
    export LOG_FOLDER="./log/"$TIME_SUFFIX"-"$SLUG
fi
mkdir -p $LOG_FOLDER
echo $LOG_FOLDER
