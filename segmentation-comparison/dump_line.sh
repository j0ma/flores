#!/usr/bin/bash

line_num=$1
split=$2

SEGM_METHODS=(
    "Reference"
    "Lowercased"
    "Moses-tokenized"
    "SentencePiece"
    "Subword-NMT"
    "LMVR"
    "MORSEL"
)

get_file_name () {
    local segm_method=$1
    local split=$2
    case $segm_method in
        "Reference")
            echo "wiki_ne_en_bpe5000_lowercase/${split}.en"
            ;;
        "Lowercased")
            echo "wiki_ne_en_bpe5000_lowercase/${split}.lower.en"
            ;;
        "Moses-tokenized")
            echo "wiki_ne_en_bpe5000_subwordnmt/${split}.en.tok.lower"
            ;;
        "SentencePiece")       
            echo "wiki_ne_en_bpe5000_lowercase/${split}.lower.bpe.en" 
            ;;
        "Subword-NMT")
            echo "wiki_ne_en_bpe5000_subwordnmt/${split}.subword-nmt.en" 
            ;;            
        "LMVR")
            echo "wiki_ne_en_lmvr-tuned/${split}.lmvr-tuned.en" 
            ;;            
        "MORSEL")
            echo "wiki_ne_en_morsel/${split}.morsel.en"
            ;;            
        *)
            echo "Improper segmentation method: ${segm_method}"
            exit 1
    esac 
}

get_nth_line () {
    local n=$1
    local file_path=$2
    head -n ${n} ${file_path} | tail -1
}

echo "Segmentation comparison:"
for method in "${SEGM_METHODS[@]}"; do
    file_path=$(get_file_name ${method} ${split})
    full_path="../data/${file_path}"
    line=$(get_nth_line ${line_num} ${full_path})
    echo "${method}: ${line}"
    echo
done

