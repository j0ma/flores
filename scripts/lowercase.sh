#!/bin/sh

# lowercase.sh

TRAIN_FOLDER=$1

if [ -z "${TRAIN_FOLDER}" ]; then
    echo "Please provide a training folder!"
    exit 1
fi

if [ -z "${MOSES_SCRIPTS}" ]; then
    echo "Please make sure the MOSES_SCRIPTS environment variable is set!"
    exit 1
fi

LOWERCASE_SCRIPT="$MOSES_SCRIPTS/tokenizer/lowercase.perl"

cd "${TRAIN_FOLDER}" || exit
echo "Now in ${TRAIN_FOLDER}"

for input_file in ./*; do
    output_file=$input_file"_new"
    $LOWERCASE_SCRIPT \
        <"${input_file}" \
        >"${output_file}"
    rm "${input_file}"
    mv "${output_file}" "${input_file}"
done
