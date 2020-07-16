#!/bin/bash

input_file=$1
output_file=$2
lang=$3

if [ "${lang}" = "en" ]; then
    perl "$MOSES_SCRIPTS/tokenizer/detokenizer.perl" \
        -q -l en \
        < "${input_file}" \
        > "${output_file}"
else
    cp "${input_file}" "${output_file}"
fi
