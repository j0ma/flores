export TIME_SUFFIX=$(date -Iminutes | sed s/":"/"-"/g)
export LOG_FOLDER="./log/"$TIME_SUFFIX
mkdir -p $LOG_FOLDER
echo $LOG_FOLDER
