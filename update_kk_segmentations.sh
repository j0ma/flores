#!/usr/bin/env bash

set -euo pipefail

# back up and update lmvr
kk_lmvr="wmt19.2500.lmvr-tuned.model.kk.tar.gz"
kk_en_lmvr="wmt19.2500.lmvr-tuned.model.kk_en.tar.gz"

for file in $kk_lmvr $kk_en_lmvr
do
    mv --verbose ./segmentation-models/${file} ./segmentation-models/${file}.bak
    cp --verbose -r ./segmentation-models-225k/${file} ./segmentation-models/${file}
done

# back up and update morsel
mv --verbose ./segmentation-models/morsel/kk_en ./segmentation-models/morsel/kk_en_bak
cp --verbose -r ./segmentation-models-225k/morsel/kk_en ./segmentation-models/morsel/kk_en
