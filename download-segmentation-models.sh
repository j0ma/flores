#!/usr/bin/sh

for ZIP_FILE in "segmentation-models.zip" "segmentation-models-225k.zip"
do
    URL_STUB="https://j0ma.keybase.pub/models/"
    wget "${URL_STUB}/${ZIP_FILE}"
    unzip "${ZIP_FILE}"
    rm "${ZIP_FILE}"
done

./update_kk_segmentations.sh
