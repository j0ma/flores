SLUG=$1
export TIME_SUFFIX=$(date -Iminutes | sed s/":"/"-"/g)
if [ -z $SLUG ]
then
    export RESULTS_FOLDER="./evaluate/"$TIME_SUFFIX
else
    export RESULTS_FOLDER="./evaluate/"$TIME_SUFFIX"-"$SLUG
fi

mkdir -p $RESULTS_FOLDER
echo $RESULTS_FOLDER
