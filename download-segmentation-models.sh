#!/usr/bin/sh

ZIP_FILE=$1
if [ -z "${ZIP_FILE}" ]; then
    ZIP_FILE="segmentation-models.zip"
fi
URL_STUB="https://j0ma.keybase.pub/models/"

wget "${URL_STUB}/${ZIP_FILE}"
unzip "${ZIP_FILE}"
rm "${ZIP_FILE}"
