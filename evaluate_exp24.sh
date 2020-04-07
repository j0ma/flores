#!/bin/sh
# ad hoc script to eval exp 24

BPE_SIZE=2500
for FOLDER in $(cat exp24.folders)
do
    SLUG=$(echo $FOLDER | grep -o "exp24-.*$")
    bash ./evaluate_manual.sh $FOLDER $SLUG $BPE_SIZE
done
