#!/bin/bash

set -exo pipefail

# Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([folder])
# ARG_OPTIONAL_SINGLE([src])
# ARG_OPTIONAL_SINGLE([tgt])
# ARG_OPTIONAL_SINGLE([bpe-num-merges])
# ARG_OPTIONAL_SINGLE([bpe-vocab-size])
# ARG_OPTIONAL_SINGLE([lmvr-vocab-size])
# ARG_OPTIONAL_SINGLE([morsel-segmentations])
# ARG_OPTIONAL_SINGLE([morsel-bpe-codes])
# ARG_OPTIONAL_BOOLEAN([tokenize])
# ARG_OPTIONAL_BOOLEAN([lowercase])
# ARG_OPTIONAL_BOOLEAN([truecase])
# ARG_OPTIONAL_BOOLEAN([filter-cyrillic-in-fi])
# ARG_OPTIONAL_BOOLEAN([filter-cyrillic-in-en])
# ARG_OPTIONAL_BOOLEAN([filter-latin-in-kk])
# ARG_OPTIONAL_BOOLEAN([subword-nmt])
# ARG_OPTIONAL_BOOLEAN([sentencepiece])
# ARG_OPTIONAL_BOOLEAN([lmvr])
# ARG_OPTIONAL_BOOLEAN([lmvr-tuned])
# ARG_OPTIONAL_BOOLEAN([morsel])
# ARG_HELP([<The general help message of my script>])
# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.8.1 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info

if [ -z "${MOSES_SCRIPTS}" ]; then
    echo "Please make sure the MOSES_SCRIPTS environment variable is set!"
    exit 1
fi

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

MOSES_TOKENIZER_SCRIPT="$MOSES_SCRIPTS/tokenizer/tokenizer.perl"
MOSES_DETOKENIZER_SCRIPT="$MOSES_SCRIPTS/tokenizer/detokenizer.perl"
MOSES_LOWERCASE_SCRIPT="$MOSES_SCRIPTS/tokenizer/lowercase.perl"
MOSES_CLEAN="$MOSES_SCRIPTS/training/clean-corpus-n.perl"
MOSES_NORM_PUNC="$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl"
MOSES_REM_NON_PRINT_CHAR="$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl"

die() {
    local _ret=$2
    test -n "$_ret" || _ret=1
    test "$_PRINT_HELP" = yes && print_help >&2
    echo "$1" >&2
    exit ${_ret}
}

begins_with_short_option() {
    local first_option all_short_options='h'
    first_option="${1:0:1}"
    test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_folder=
_arg_src=
_arg_tgt=
_arg_bpe_num_merges=
_arg_bpe_vocab_size=
_arg_lmvr_vocab_size=
_arg_morsel_segmentations=
_arg_morsel_bpe_codes=
_arg_tokenize="off"
_arg_lowercase="off"
_arg_truecase="off"
_arg_filter_cyrillic_in_fi="off"
_arg_filter_cyrillic_in_en="off"
_arg_filter_latin_in_kk="off"
_arg_subword_nmt="off"
_arg_sentencepiece="off"
_arg_lmvr="off"
_arg_lmvr_tuned="off"
_arg_morsel="off"

print_help() {
    printf '%s\n' "<The general help message of my script>"
    printf 'Usage: %s [--folder <arg>] [--src <arg>] [--tgt <arg>] [--bpe-num-merges <arg>] [--bpe-vocab-size <arg>] [--lmvr-vocab-size <arg>] [--morsel-segmentations <arg>] [--morsel-bpe-codes <arg>] [--(no-)tokenize] [--(no-)lowercase] [--(no-)truecase] [--(no-)filter-cyrillic-in-fi] [--(no-)filter-cyrillic-in-en] [--(no-)filter-latin-in-kk] [--(no-)subword-nmt] [--(no-)sentencepiece] [--(no-)lmvr] [--(no-)lmvr-tuned] [--(no-)morsel] [-h|--help]\n' "$0"
    printf '\t%s\n' "-h, --help: Prints help"
}

parse_commandline() {
    while test $# -gt 0; do
        _key="$1"
        case "$_key" in
        --folder)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_folder="$2"
            shift
            ;;
        --folder=*)
            _arg_folder="${_key##--folder=}"
            ;;
        --src)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_src="$2"
            shift
            ;;
        --src=*)
            _arg_src="${_key##--src=}"
            ;;
        --tgt)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_tgt="$2"
            shift
            ;;
        --tgt=*)
            _arg_tgt="${_key##--tgt=}"
            ;;
        --bpe-num-merges)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_bpe_num_merges="$2"
            shift
            ;;
        --bpe-num-merges=*)
            _arg_bpe_num_merges="${_key##--bpe-num-merges=}"
            ;;
        --bpe-vocab-size)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_bpe_vocab_size="$2"
            shift
            ;;
        --bpe-vocab-size=*)
            _arg_bpe_vocab_size="${_key##--bpe-vocab-size=}"
            ;;
        --lmvr-vocab-size)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_lmvr_vocab_size="$2"
            shift
            ;;
        --lmvr-vocab-size=*)
            _arg_lmvr_vocab_size="${_key##--lmvr-vocab-size=}"
            ;;
        --morsel-segmentations)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_morsel_segmentations="$2"
            shift
            ;;
        --morsel-segmentations=*)
            _arg_morsel_segmentations="${_key##--morsel-segmentations=}"
            ;;
        --morsel-bpe-codes)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_morsel_bpe_codes="$2"
            shift
            ;;
        --morsel-bpe-codes=*)
            _arg_morsel_bpe_codes="${_key##--morsel-bpe-codes=}"
            ;;
        --no-tokenize | --tokenize)
            _arg_tokenize="on"
            test "${1:0:5}" = "--no-" && _arg_tokenize="off"
            ;;
        --no-lowercase | --lowercase)
            _arg_lowercase="on"
            test "${1:0:5}" = "--no-" && _arg_lowercase="off"
            ;;
        --no-truecase | --truecase)
            _arg_truecase="on"
            test "${1:0:5}" = "--no-" && _arg_truecase="off"
            ;;
        --no-filter-cyrillic-in-fi | --filter-cyrillic-in-fi)
            _arg_filter_cyrillic_in_fi="on"
            test "${1:0:5}" = "--no-" && _arg_filter_cyrillic_in_fi="off"
            ;;
        --no-filter-cyrillic-in-en | --filter-cyrillic-in-en)
            _arg_filter_cyrillic_in_en="on"
            test "${1:0:5}" = "--no-" && _arg_filter_cyrillic_in_en="off"
            ;;
        --no-filter-latin-in-kk | --filter-latin-in-kk)
            _arg_filter_latin_in_kk="on"
            test "${1:0:5}" = "--no-" && _arg_filter_latin_in_kk="off"
            ;;
        --no-subword-nmt | --subword-nmt)
            _arg_subword_nmt="on"
            test "${1:0:5}" = "--no-" && _arg_subword_nmt="off"
            ;;
        --no-sentencepiece | --sentencepiece)
            _arg_sentencepiece="on"
            test "${1:0:5}" = "--no-" && _arg_sentencepiece="off"
            ;;
        --no-lmvr | --lmvr)
            _arg_lmvr="on"
            test "${1:0:5}" = "--no-" && _arg_lmvr="off"
            ;;
        --no-lmvr-tuned | --lmvr-tuned)
            _arg_lmvr_tuned="on"
            test "${1:0:5}" = "--no-" && _arg_lmvr_tuned="off"
            ;;
        --no-morsel | --morsel)
            _arg_morsel="on"
            test "${1:0:5}" = "--no-" && _arg_morsel="off"
            ;;
        -h | --help)
            print_help
            exit 0
            ;;
        -h*)
            print_help
            exit 0
            ;;
        *)
            _PRINT_HELP=yes die "FATAL ERROR: Got an unexpected argument '$1'" 1
            ;;
        esac
        shift
    done
}

parse_commandline "$@"

# OTHER STUFF GENERATED BY Argbash

### END OF CODE GENERATED BY Argbash (sortof) ### ])
# [ <-- needed because of Argbash

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
INDIC_TOKENIZER="$SCRIPTS/indic_norm_tok.sh"

tokenization_pipeline() {

    INPUT_FILE=$1
    OUTPUT_FILE=$2
    LANGUAGE=$3
    if [ "${LANGUAGE}" = "kk" ]; then
        LANGUAGE="ru"
    fi

    if [ "${LANGUAGE}" = "gu" ]; then
        file "${INDIC_TOKENIZER//bash /}" || exit
        file "${INPUT_FILE}" || exit
        bash "${INDIC_TOKENIZER}" "${LANGUAGE}" "${INPUT_FILE}" >"${OUTPUT_FILE}"
    else
        cat "${INPUT_FILE}" |
            sed "s/--/ -- /g" |
            perl "${MOSES_NORM_PUNC}" "${LANGUAGE}" |
            perl "${MOSES_REM_NON_PRINT_CHAR}" |
            perl "${MOSES_TOKENIZER_SCRIPT}" -l "${LANGUAGE}" |
            perl -C -MHTML::Entities -pe 'decode_entities($_);' \
                >"${OUTPUT_FILE}"
    fi
}

convert_lowercase() {

    INPUT_FILE=$1
    OUTPUT_FILE=$2
    "$MOSES_LOWERCASE_SCRIPT" \
        <"$INPUT_FILE" >"$OUTPUT_FILE"
}

TRAIN_MINLEN=1   # remove sentences with <1 BPE token
TRAIN_MAXLEN=250 # remove sentences with >250 BPE tokens

printf 'Value of --%s: %s\n' 'folder' "$_arg_folder"
printf 'Value of --%s: %s\n' 'src' "${_arg_src}"
printf 'Value of --%s: %s\n' 'tgt' "${_arg_tgt}"
printf 'Value of --%s: %s\n' 'bpe-num-merges' "$_arg_bpe_num_merges"
printf 'Value of --%s: %s\n' 'bpe-vocab-size' "$_arg_bpe_vocab_size"
printf 'Value of --%s: %s\n' 'lmvr-vocab-size' "$_arg_lmvr_vocab_size"
printf 'Value of --%s: %s\n' 'morsel-segmentations' "$_arg_morsel_segmentations"
printf 'Value of --%s: %s\n' 'morsel-bpe-codes' "$_arg_morsel_bpe_codes"
printf "'%s' is %s\\n" 'tokenize' "$_arg_tokenize"
printf "'%s' is %s\\n" 'lowercase' "$_arg_lowercase"
printf "'%s' is %s\\n" 'truecase' "$_arg_truecase"
printf "'%s' is %s\\n" 'filter-cyrillic-in-fi' "$_arg_filter_cyrillic_in_fi"
printf "'%s' is %s\\n" 'filter-cyrillic-in-en' "$_arg_filter_cyrillic_in_en"
printf "'%s' is %s\\n" 'filter-latin-in-kk' "$_arg_filter_latin_in_kk"
printf "'%s' is %s\\n" 'subword-nmt' "$_arg_subword_nmt"
printf "'%s' is %s\\n" 'sentencepiece' "$_arg_sentencepiece"
printf "'%s' is %s\\n" 'lmvr' "$_arg_lmvr"
printf "'%s' is %s\\n" 'lmvr-tuned' "$_arg_lmvr_tuned"
printf "'%s' is %s\\n" 'morsel' "$_arg_morsel"

# take note of the foreign language
for lang in "${_arg_src}" "${_arg_tgt}"; do
    if [ "${lang}" = "en" ]; then
        continue
    fi
    foreign=$lang
done

interim_data_folder="${_arg_folder}/${foreign}-en/interim/"
final_data_folder="${_arg_folder}/${foreign}-en/final/"

for split in "train" "dev" "test"; do
    for lang in "${_arg_src}" "${_arg_tgt}"; do

        suffix=""
        interim_path="${interim_data_folder}/${split}"
        final_path="${final_data_folder}/${split}"
        if [ ! "${split}" = "train" ]; then
            output_fname="${split}.${_arg_src}${_arg_tgt}.${lang}"
        else
            output_fname="${split}.${lang}"
        fi
        if [ "${_arg_tokenize}" = "on" ]; then
            suffix="${suffix}.tok"
            new_output_fname="${output_fname}.tok"
            tokenization_pipeline \
                "${interim_path}/${output_fname}" \
                "${interim_path}/${new_output_fname}" \
                "${lang}"
            #output_fname="${output_fname}${suffix}"
        fi

        if [ "${_arg_lowercase}" = "on" ]; then
            suffix="${suffix}.lower"
            new_output_fname="${new_output_fname}.lower"
            convert_lowercase \
                "${interim_path}/${output_fname}" \
                "${interim_path}/${new_output_fname}"
            #output_fname="${output_fname}${suffix}"
        fi

    done
done

if [ "${_arg_subword_nmt}" = "on" ]; then

    echo "subword-nmt detected!"

    test -z "${_arg_bpe_num_merges}" &&
        echo "Please provide number of BPE merges!" && exit 1

    model_name="subword-nmt"
    data_bin_folder="data-bin/wmt19-${model_name}/${foreign}-en/${_arg_src}-${_arg_tgt}"
    mkdir -p "${data_bin_folder}"

    # concatenate training sets to one big file
    rm -f "${interim_data_folder}/train/train.all${suffix}"
    cat "${interim_data_folder}/train/"train.*"${suffix}" \
        >"${interim_data_folder}/train/train.all${suffix}"

    # perform bpe training without outputting segmentation
    segm_input_file="${interim_data_folder}/train/train.all${suffix}"
    joint_codes_file="${interim_data_folder}/train/subword-nmt.codes"

    bash "$SCRIPTS/segment.sh" \
        --input "${segm_input_file}" \
        --output "none" \
        --model "${model_name}" \
        --model-binary none \
        --bpe-size "${_arg_bpe_num_merges}" \
        --codes "${joint_codes_file}" \
        --lang "foo"

    # apply bpe and output segmentation
    for split in "train" "dev" "test"; do
        for lang in "${_arg_src}" "${_arg_tgt}"; do

            input_stub="${interim_data_folder}/$split/$split"
            output_stub=${final_data_folder}/$split/$split.${model_name}
            if [ ! "${split}" = "train" ]; then
                input_stub="${input_stub}.${_arg_src}${_arg_tgt}"
                output_stub="${output_stub}.${_arg_src}${_arg_tgt}"
            fi
            segm_input_file="${input_stub}.${lang}${suffix}"
            segm_output_file=${output_stub}.${lang}
            bash "$SCRIPTS/segment.sh" \
                --input "${segm_input_file}" \
                --output "${segm_output_file}" \
                --model "${model_name}" \
                --model-binary none \
                --bpe-size "${_arg_bpe_num_merges}" \
                --codes "${joint_codes_file}" \
                --lang "${lang}"
        done
    done

    # binarize data
    stub=".${_arg_src}${_arg_tgt}"
    fairseq-preprocess \
        --source-lang ${_arg_src} --target-lang ${_arg_tgt} \
        --trainpref ${final_data_folder}/train/train.${model_name} \
        --validpref ${final_data_folder}/dev/dev.${model_name}${stub} \
        --testpref ${final_data_folder}/test/test.${model_name}${stub} \
        --destdir ${data_bin_folder} \
        --joined-dictionary \
        --workers 8

elif [ "${_arg_sentencepiece}" = "on" ]; then

    test -z "${_arg_bpe_vocab_size}" &&
        echo "Please provide BPE vocab size!" && exit 1

    model_name="sentencepiece"
    data_bin_folder="data-bin/wmt19-${model_name}/${foreign}-en/${_arg_src}-${_arg_tgt}"
    mkdir -p "${data_bin_folder}"

    # learn BPE with sentencepiece
    python $SPM_TRAIN \
        --input=${interim_data_folder}/train/train.${_arg_src},${interim_data_folder}/train/train.${_arg_tgt} \
        --model_prefix=${data_bin_folder}/sentencepiece.bpe \
        --vocab_size="$_arg_bpe_vocab_size" \
        --character_coverage=1.0 \
        --model_type=bpe

    # encode train/dev/test
    for split in "train" "dev" "test"; do

        input_stub="${interim_data_folder}/$split/$split"
        output_stub=${final_data_folder}/$split/$split.${model_name}
        if [ ! "${split}" = "train" ]; then
            input_stub="${input_stub}.${_arg_src}${_arg_tgt}"
            output_stub="${output_stub}.${_arg_src}${_arg_tgt}"
        fi
        python $SPM_ENCODE \
            --model ${data_bin_folder}/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs ${input_stub}.${_arg_src} ${input_stub}.${_arg_tgt} \
            --outputs ${output_stub}.${_arg_src} ${output_stub}.${_arg_tgt}
    done

    # binarize data
    stub=".${_arg_src}${_arg_tgt}"
    fairseq-preprocess \
        --source-lang ${_arg_src} --target-lang ${_arg_tgt} \
        --trainpref ${final_data_folder}/train/train.${model_name} \
        --validpref ${final_data_folder}/dev/dev.${model_name}${stub} \
        --testpref ${final_data_folder}/test/test.${model_name}${stub} \
        --destdir ${data_bin_folder} \
        --joined-dictionary \
        --workers 8

elif [ "${_arg_lmvr}" = "on" ]; then
    echo "LMVR not implemented" && exit 1
elif [ "${_arg_lmvr_tuned}" = "on" ]; then
    #echo "LMVR-tuned not implemented" && exit 1

    model_name="lmvr-tuned"
    segm_model_folder=$ROOT/segmentation-models/

    data_bin_folder="data-bin/wmt19-${model_name}/${foreign}-en/${_arg_src}-${_arg_tgt}"

    # activate virtual environment
    echo "activating LMVR virtual environment..."
    if [ -z "$LMVR_ENV_PATH" ]; then
        source "$(pwd)/scripts/lmvr-environment-variables.sh"
    fi
    source "$LMVR_ENV_PATH/bin/activate"

    # make sure we're actually running 2.7
    if [ -z "$(python -c "import sys; print(sys.version)" | grep -E "^2\.7")" ]; then
        echo "Need to be running Python 2.7 for LMVR!"
        exit 1
    fi

    for split in "train" "dev" "test"; do
        for lang in "${_arg_src}" "${_arg_tgt}"; do

            echo "Check python version"
            which python
            python --version

            input_stub="${interim_data_folder}/$split/$split"
            output_stub=${final_data_folder}/$split/$split.${model_name}
            if [ ! "${split}" = "train" ]; then
                input_stub="${input_stub}.${_arg_src}${_arg_tgt}"
                output_stub="${output_stub}.${_arg_src}${_arg_tgt}"
            fi
            segm_input_file="${input_stub}.${lang}${suffix}"
            segm_output_file=${output_stub}.${lang}

            if [ "${lang}" = "en" ]; then
                lang_alias="${foreign}_en"
            else
                lang_alias="${lang}"
            fi
            lmvr_model_file="${segm_model_folder}/wmt19.2500.lmvr-tuned.model.${lang_alias}.tar.gz"
            bash "$SCRIPTS/segment.sh" \
                --input "${segm_input_file}" \
                --output "${segm_output_file}" \
                --model "${model_name}" \
                --model-binary "${lmvr_model_file}" \
                --lang "${lang}" \
                --kind "${split}"
        done
    done

    # deactivate the environment
    deactivate

    echo "Done! Time to binarize the data..."

    # binarize data
    stub=".${_arg_src}${_arg_tgt}"
    fairseq-preprocess \
        --source-lang ${_arg_src} --target-lang ${_arg_tgt} \
        --trainpref ${final_data_folder}/train/train.${model_name} \
        --validpref ${final_data_folder}/dev/dev.${model_name}${stub} \
        --testpref ${final_data_folder}/test/test.${model_name}${stub} \
        --destdir ${data_bin_folder} \
        --joined-dictionary \
        --workers 8

elif [ "${_arg_morsel}" = "on" ]; then

    #echo "MORSEL not implmemented" && exit 1

    echo "MORSEL from Lignos (2010) ..."
    model_name="morsel"
    data_bin_folder="data-bin/wmt19-${model_name}/${foreign}-en/${_arg_src}-${_arg_tgt}"
    mkdir -p "${data_bin_folder}"

    for split in "train" "dev" "test"; do
        for lang in "$_arg_src" "$_arg_tgt"; do

            echo "Processing ${split} set for ${lang}"
            echo "Actual segmentation..."
            input_stub="${interim_data_folder}/${split}/${split}"
            output_stub=${final_data_folder}/${split}/${split}.${model_name}
            if [ ! "${split}" = "train" ]; then
                input_stub="${input_stub}.${_arg_src}${_arg_tgt}"
                output_stub="${output_stub}.${_arg_src}${_arg_tgt}"
            fi
            segm_input_file="${input_stub}.${lang}${suffix}"
            segm_output_file=${output_stub}.${lang}
            morsel_root="./segmentation-models/morsel/${foreign}_en/${lang}/"
            bash ./scripts/segment_using_morsel.sh \
                --sentences "${segm_input_file}" \
                --morsel-segmentations "${morsel_root}/morsel_seg_bpe_map.txt" \
                --bpe-codes "${morsel_root}/stem_code.txt" \
                --output-file "${segm_output_file}"
        done
    done

    stub=".${_arg_src}${_arg_tgt}"
    fairseq-preprocess \
        --source-lang ${_arg_src} --target-lang ${_arg_tgt} \
        --trainpref ${final_data_folder}/train/train.${model_name} \
        --validpref ${final_data_folder}/dev/dev.${model_name}${stub} \
        --testpref ${final_data_folder}/test/test.${model_name}${stub} \
        --destdir ${data_bin_folder} \
        --joined-dictionary \
        --workers 8
fi
