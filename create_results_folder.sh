export TIME_SUFFIX=$(date -Iminutes | sed s/":"/"-"/g)
export RESULTS_FOLDER="./evaluate/"$TIME_SUFFIX
mkdir -p $RESULTS_FOLDER
echo $RESULTS_FOLDER
