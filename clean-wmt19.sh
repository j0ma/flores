#!/bin/bash

set -eo pipefail

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

moses_pipeline() {

    INPUT_FILE=$1
    OUTPUT_FILE=$2
    LANGUAGE=$3
    if [ "${LANGUAGE}" = "kk" ]; then
        LANGUAGE="ru"
    fi

    cat "$INPUT_FILE" |
        sed "s/--/ -- /g" |
        perl "$MOSES_NORM_PUNC" "$LANGUAGE" |
        perl "$MOSES_REM_NON_PRINT_CHAR" |
        perl "$MOSES_TOKENIZER_SCRIPT" -l "${LANGUAGE}" |
        perl -C -MHTML::Entities -pe 'decode_entities($_);' \
            >"$OUTPUT_FILE"
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
printf 'Value of --%s: %s\n' 'src' "$_arg_src"
printf 'Value of --%s: %s\n' 'tgt' "$_arg_tgt"
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
for lang in "${_arg_src}" "${_arg_tgt}"
do 
    if [ "${lang}" = "en" ]; then
        continue
    fi
    foreign=$lang
done

interim_data_folder="${_arg_folder}/${foreign}-en/interim/"
final_data_folder="${_arg_folder}/${foreign}-en/final/"
for split in "train" "dev" "test"
do 
    for lang in "${_arg_src}" "${_arg_tgt}"
    do
        interim_path="${interim_data_folder}/${split}"
        final_path="${final_data_folder}/${split}"
        output_fname="${split}.${lang}"
        if [ "${_arg_tokenize}" = "on" ]; then
            moses_pipeline \
                "${interim_path}/${output_fname}" \
                "${interim_path}/${output_fname}.tok" \
                "${lang}"
            output_fname="${output_fname}.tok"
        fi

        if [ "${_arg_lowercase}" = "on" ]; then
            convert_lowercase \
                "${interim_path}/${output_fname}" \
                "${interim_path}/${output_fname}.lower"
            output_fname="${output_fname}.lower"
        fi

        # finally output everything to /final
        cp -v  "${interim_path}/${output_fname}" \
            "${final_path}/${output_fname}"
    done
done


# ] <-- needed because of Argbash
