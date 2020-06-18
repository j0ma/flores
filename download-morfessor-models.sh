#!/usr/bin/sh

TMP_BIN=./morfessor-models/
mkdir -p $TMP_BIN

EN_MODEL="all-flores-words-en-morfessor-baseline-batch-recursive-en.bin"
NE_MODEL="all-flores-words-ne-morfessor-baseline-batch-recursive-ne.bin"
SI_MODEL="all-flores-words-si-morfessor-baseline-batch-recursive-si.bin"

cd $TMP_BIN
for model in $EN_MODEL $NE_MODEL $SI_MODEL;
do
    wget https://j0ma.keybase.pub/models/$model
done
cd ..
