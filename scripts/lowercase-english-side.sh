#!/bin/sh

# lowercase-english-side.sh
# NOTE: as of 6/22/20 this is not used

# only included here for archival purposes as we ran one experiment with it

TRAIN_FOLDER=$1

if [ -z "${TRAIN_FOLDER}" ]
then
    echo "Please provide a training folder!"
    exit 1
fi

cd "${TRAIN_FOLDER}" || exit

for input_file in ./*.en
do
    output_file=$input_file"_new"
    awk '{ print tolower($0) }' \
        < "${input_file}" \
        > "${output_file}"
    rm "${input_file}"
    mv "${output_file}" "${input_file}"
done
