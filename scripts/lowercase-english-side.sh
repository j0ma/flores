#!/bin/sh

# lowercase-english-side.sh

TRAIN_FOLDER=$1

if [ -z "${TRAIN_FOLDER}" ]
then
    echo "Please provide a training folder!"
    exit 1
fi

cd "${TRAIN_FOLDER}" || exit

for input_file in ./*.en
do
    output_file=$(echo "${input_file}" | sed "s/\.en$/\.lower.en/g")
    echo "${input_file} -> ${output_file}"
    awk '{ print tolower($0) }' \
        < "${input_file}" \
        > "${output_file}"
done
