#!/bin/bash

system_output=$1
src=$2
tgt=$3

grep ^S $system_output | cut -f 2 > ${system_output}.${src}
grep ^H $system_output | cut -f 3 > ${system_output}.${tgt}
perl $MOSES_SCRIPTS/tokenizer/detokenizer.perl \
    -q -l en \
    < ${system_output}.${tgt} \
    > ${system_output}.detok.${tgt}
