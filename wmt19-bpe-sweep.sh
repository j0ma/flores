#!/bin/bash

set -euo pipefail

src=$1
tgt=$2
gpu_device=$3

# detect foreign language
for lang in "${src}" "${tgt}"; do
    [ "${lang}" = "en" ] && continue
    foreign=$lang
done

echo "Foreign language detected: ${foreign}"

BPE_SIZES=(1000 2500 7500 10000)

train () {
    for bpe_size in "${BPE_SIZES[@]}"; do

        # train subword-nmt
        bash ./train-wmt19.sh \
            --src "${src}" --tgt "${tgt}" \
            --from-seed 10 --to-seed 14 \
            --bpe-size "${bpe_size}" --cuda-device "${gpu_device}" \
            --model-name subword-nmt \
            --clip-norm 0.1 \
            --checkpoint-dir "auto" \
            --log-dir "auto" \
            --data-dir "data-bin/wmt19-bpe${bpe_size}-subword-nmt/${foreign}-en/${src}-${tgt}/" \
            --fp16 --slug "wmt19-${src}${tgt}-bpe${bpe_size}-subword-nmt-sweep"

        # eval subword-nmt
        bash ./eval-wmt19.sh \
            --checkpoint-glob "./checkpoints/*wmt19-${src}${tgt}-bpe${bpe_size}-subword-nmt-sweep*" \
            --src "${src}" --tgt "${tgt}" --eval-on "test" \
            --data-folder "data/wmt19-bpe${bpe_size}/${foreign}-en/final/test" \
            --data-bin-folder "data-bin/wmt19-bpe${bpe_size}-subword-nmt/${foreign}-en/${src}-${tgt}" \
            --segmentation-model-type "subword-nmt" \
            --model-name "subword-nmt" \
            --reference "./data/wmt19-bpe${bpe_size}/${foreign}-en/interim/test/test.${src}${tgt}.${tgt}" \
            --remove-bpe-type "regular" \
            --translation-output-folder "./translation-output-wmt19-bpe${bpe_size}"

        # train sentencepiece
        bash ./train-wmt19.sh \
            --src "${src}" --tgt "${tgt}" \
            --from-seed 10 --to-seed 14 \
            --bpe-size "${bpe_size}" --cuda-device "${gpu_device}" \
            --model-name sentencepiece \
            --clip-norm 0.1 \
            --checkpoint-dir "auto" \
            --log-dir "auto" \
            --data-dir "data-bin/wmt19-bpe${bpe_size}-sentencepiece/${foreign}-en/${src}-${tgt}/" \
            --fp16 --slug "wmt19-${src}${tgt}-bpe${bpe_size}-sentencepiece-sweep"

        # eval sentencepiece
        bash ./eval-wmt19.sh \
            --checkpoint-glob "./checkpoints/*wmt19-${src}${tgt}-bpe${bpe_size}sentencepiece-sweep*" \
            --src "${src}" --tgt "${tgt}" --eval-on "test" \
            --data-folder "data/wmt19-bpe${bpe_size}/${foreign}-en/final/test" \
            --data-bin-folder "data-bin/wmt19-bpe${bpe_size}-sentencepiece/${foreign}-en/${src}-${tgt}" \
            --segmentation-model-type "sentencepiece" \
            --model-name "baseline" \
            --reference "./data/wmt19-bpe${bpe_size}/${foreign}-en/interim/test/test.${src}${tgt}.${tgt}" \
            --remove-bpe-type "sentencepiece" \
            --translation-output-folder "./translation-output-wmt19-bpe${bpe_size}"

    done
}

train
