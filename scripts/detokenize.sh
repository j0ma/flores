#!/bin/bash

input_file=$1
output_file=$2
lang=$3

if [ "${lang}" = "en" ]; then
    LANG="en"
elif [ "${lang}" = "kk" ]; then
    LANG="ru"
else
    cp "${input_file}" "${output_file}"
    exit
fi

perl "$MOSES_SCRIPTS/tokenizer/detokenizer.perl" \
    -q -l "${LANG}" \
    < "${input_file}" \
    > "${output_file}"
