#!/bin/bash

# NOTE: THIS SCRIPT USES ARGBASH TO SET UP COMMAND LINE ARGUMENTS

# Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([input])
# ARG_OPTIONAL_SINGLE([output])
# ARG_OPTIONAL_SINGLE([model])
# ARG_OPTIONAL_SINGLE([model-binary])
# ARG_OPTIONAL_SINGLE([bpe-size])
# ARG_HELP([<The general help message of my script>])
# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.8.1 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info

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
_arg_input=
_arg_output=
_arg_model=
_arg_model_binary=
_arg_bpe_size=

print_help() {
    USAGE_MSG="""
    Usage: bash segment.sh --input <untokenized input file> 
                           --output <tokenized output file> 
                           --model <type of model we are using>
                           --model-binary <path to model binary>
                           [--bpe-size <bpe size>]
"""
    printf '\n%s\n' "segment.sh -- a script for subword segmentation using Morfessor / Flatcat / LMVR / subword-nmt"
    echo "$USAGE_MSG"
}

parse_commandline() {
    while test $# -gt 0; do
        _key="$1"
        case "$_key" in
        --input)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_input="$2"
            shift
            ;;
        --input=*)
            _arg_input="${_key##--input=}"
            ;;
        --output)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_output="$2"
            shift
            ;;
        --output=*)
            _arg_output="${_key##--output=}"
            ;;
        --model)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_model="$2"
            shift
            ;;
        --model=*)
            _arg_model="${_key##--model=}"
            ;;
        --model-binary)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_model_binary="$2"
            shift
            ;;
        --model-binary=*)
            _arg_model_binary="${_key##--model-binary=}"
            ;;
        --bpe-size)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_bpe_size="$2"
            shift
            ;;
        --bpe-size=*)
            _arg_bpe_size="${_key##--bpe-size=}"
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

# Validate the command line arguments that were passed in
validate() {
    [ -z "$1" ] && print_help && exit 1
}

for var in "$_arg_input" "$_arg_output" "$_arg_model" "$_arg_model_binary"; do
    validate "$var"
done

ROOT=$(dirname "$0")
MF_SEGMENT_COMMAND="python $ROOT/segment-sentences-morfessor.py"

# Set up the segmentation functions
segment_morfessor_baseline() {
    echo "Segmenting with Morfessor Baseline..."
    INPUT_FILE=$1
    MODEL_BINARY=$2
    OUTPUT_FILE=$3

    $MF_SEGMENT_COMMAND \
        -i "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        -m "$MODEL_BINARY" \
        --lang foo
}

segment_flatcat() {

    INPUT_FILE=$1
    MODEL_BINARY=$2
    OUTPUT_FILE=$3

    echo "Segmenting with Flatcat..."
    $MF_SEGMENT_COMMAND \
        -i "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        -m "$MODEL_BINARY" \
        --lang foo
}

segment_subword_nmt() {

    INPUT_FILE=$1
    OUTPUT_FILE=$3
    CODES_FILE=$3.codes
    BPE_SIZE=$2
    LANGUAGE=$4

    IS_TRAIN=$(echo $INPUT_FILE | grep "train\.")
    if [ ! -z "$IS_TRAIN" ]; then
        echo "Training set detected!"
        echo "BPE Size: $BPE_SIZE"
        echo "Input: $INPUT_FILE"
        echo "Codes: $CODES_FILE"
        echo "Output: $OUTPUT_FILE"
        echo "Learning BPE with subword-nmt..."

        subword-nmt learn-bpe \
            -t -s "$BPE_SIZE" \
            <"$INPUT_FILE" \
            >"$CODES_FILE"
    else
        echo "Not a training set!"
        echo "Not learning BPE..."

        # we need to grab the correct codes file
        CODES_FILE=$(
            echo "$CODES_FILE" |
                sed "s/valid/train/g" |
                sed "s/test/train/g"
        )

        echo "BPE Size: $BPE_SIZE"
        echo "Input: $INPUT_FILE"
        echo "Codes: $CODES_FILE"
        echo "Output: $OUTPUT_FILE"
    fi

    echo "Applying BPE with subword-nmt..."
    subword-nmt apply-bpe \
        -c "$CODES_FILE" \
        <"$INPUT_FILE" \
        >"$OUTPUT_FILE"
}

segment_lmvr() {
    echo "This is segment_lmvr"
    echo "Implement me!"
    exit 1
}

# Perform segmentation with the correct model
case "$_arg_model" in
baseline)
    segment_morfessor_baseline \
        "$_arg_input" \
        "$_arg_model_binary" \
        "$_arg_output"
    ;;
flatcat)
    segment_flatcat \
        "$_arg_input" \
        "$_arg_model_binary" \
        "$_arg_output"
    ;;
lmvr)
    segment_lmvr \
        "$_arg_input" \
        "$_arg_model_binary" \
        "$_arg_output"
    ;;
subword-nmt)
    validate "$_arg_bpe_size"
    segment_subword_nmt \
        "$_arg_input" \
        "$_arg_bpe_size" \
        "$_arg_output"
    ;;
*)
    _PRINT_HELP=yes die "FATAL ERROR: Got an unexpected model type '$_arg_model'. Supported: baseline, flatcat, lmvr, subword-nmt" 1
    ;;
esac

# ] <-- needed because of Argbash
