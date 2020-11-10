BPE_SIZES=(1000 2500 7500 10000)

for bpe_size in "${BPE_SIZES[@]}"; do
    data_folder_path="$(pwd)/data/wmt19-bpe${bpe_size}"
    echo "Downloading WMT19 data for GU-EN and KK-EN"
    echo "Saving data to ${data_folder_path}"
    #bash ./download-wmt19.sh \
        #--kk-en --gu-en \
        #--output-folder "${data_folder_path}"
    bash convert-sent-per-line-wmt19.sh \
        --output-folder "${data_folder_path}" \
        --gu-en --kk-en

    echo "About to tokenize & segment..."
    for foreign in "kk" "gu"; do
        for src in "en" "${foreign}"; do
            for tgt in "en" "${foreign}"; do
                [ "${src}" = "${tgt}"  ] && continue
                echo "Segmenting ${src}-${tgt} using SentencePiece..."
                bash tokenize-segment-wmt19.sh \
                    --src "${src}" --tgt "${tgt}" \
                    --lowercase --tokenize \
                    --sentencepiece \
                    --bpe-vocab-size ${bpe_size} \
                    --folder "${data_folder_path}"
                
                echo "Segmenting ${src}-${tgt} using Subword-NMT..."
                bash tokenize-segment-wmt19.sh \
                    --src "${src}" --tgt "${tgt}" \
                    --lowercase --tokenize \
                    --subword-nmt \
                    --bpe-num-merges "${bpe_size}" \
                    --folder "${data_folder_path}"
            done
        done
    done

done


