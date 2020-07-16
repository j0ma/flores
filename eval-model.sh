#!/bin/bash

# Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([src])
# ARG_OPTIONAL_SINGLE([tgt])
# ARG_OPTIONAL_SINGLE([eval-on])
# ARG_OPTIONAL_SINGLE([data-folder])
# ARG_OPTIONAL_SINGLE([data-bin-folder])
# ARG_OPTIONAL_SINGLE([model-checkpoint])
# ARG_OPTIONAL_SINGLE([model-type])
# ARG_OPTIONAL_SINGLE([output-file])
# ARG_OPTIONAL_SINGLE([reference])
# ARG_OPTIONAL_SINGLE([remove-bpe])

# ARG_HELP([<The general help message of my script>])
# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.8.1 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info

set -eo pipefail

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
_arg_src=
_arg_tgt=
_arg_eval_on=
_arg_data_folder=
_arg_data_bin_folder=
_arg_model_checkpoint=
_arg_model_type=
_arg_output_file=
_arg_reference=
_arg_remove_bpe="regular"

print_help() {
    printf '%s\n' "<The general help message of my script>"
    printf 'Usage: %s [--src <arg>] [--tgt <arg>] [--eval-on <arg>] [--data-folder <arg>] [--data-bin-folder <arg>] [--model-checkpoint <arg>] [--output-file <arg>] [--reference <arg>] [--remove-bpe <arg>] [-h|--help]\n' "$0"
    printf '\t%s\n' "-h, --help: Prints help"
}

parse_commandline() {
    while test $# -gt 0; do
        _key="$1"
        case "$_key" in
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
        --eval-on)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_eval_on="$2"
            shift
            ;;
        --eval-on=*)
            _arg_eval_on="${_key##--eval-on=}"
            ;;
        --data-folder)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_data_folder="$2"
            shift
            ;;
        --data-folder=*)
            _arg_data_folder="${_key##--data-folder=}"
            ;;
        --data-bin-folder)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_data_bin_folder="$2"
            shift
            ;;
        --data-bin-folder=*)
            _arg_data_bin_folder="${_key##--data-bin-folder=}"
            ;;
        --model-checkpoint)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_model_checkpoint="$2"
            shift
            ;;
        --model-checkpoint=*)
            _arg_model_checkpoint="${_key##--model-checkpoint=}"
            ;;
        --model-type)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_model_type="$2"
            shift
            ;;
        --model-type=*)
            _arg_model_type="${_key##--model-type=}"
            ;;
        --output-file)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_output_file="$2"
            shift
            ;;
        --output-file=*)
            _arg_output_file="${_key##--output-file=}"
            ;;
        --reference)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_reference="$2"
            shift
            ;;
        --reference=*)
            _arg_reference="${_key##--reference=}"
            ;;
        --remove-bpe)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_remove_bpe="$2"
            shift
            ;;
        --remove-bpe=*)
            _arg_remove_bpe="${_key##--remove-bpe=}"
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

printf 'Value of --%s: %s\n' 'src' "$_arg_src"
printf 'Value of --%s: %s\n' 'tgt' "$_arg_tgt"
printf 'Value of --%s: %s\n' 'eval-on' "$_arg_eval_on"
printf 'Value of --%s: %s\n' 'data-folder' "$_arg_data_folder"
printf 'Value of --%s: %s\n' 'data-bin-folder' "$_arg_data_bin_folder"
printf 'Value of --%s: %s\n' 'model-checkpoint' "$_arg_model_checkpoint"
printf 'Value of --%s: %s\n' 'model-type' "$_arg_model_type"
printf 'Value of --%s: %s\n' 'output-file' "$_arg_output_file"
printf 'Value of --%s: %s\n' 'reference' "$_arg_reference"
printf 'Value of --%s: %s\n' 'remove-bpe' "$_arg_remove_bpe"

ROOT="$(pwd)"
SCRIPTS="${ROOT}/scripts"

echo "Evaluating model checkpoint...."
echo "Remove BPE: ${_arg_remove_bpe}"
# evaluate model checkpoint & create log
bash "$SCRIPTS/eval-fairseq-interactive.sh" \
    --src "${_arg_src}" --tgt "${_arg_tgt}" \
    --eval-on "${_arg_eval_on}" \
    --data-folder "${_arg_data_folder}" \
    --data-bin-folder "${_arg_data_bin_folder}" \
    --model-checkpoint "${_arg_model_checkpoint}" \
    --model-type "${_arg_model_type}" \
    --output-file "${_arg_output_file}" \
    --remove-bpe "${_arg_remove_bpe}"

echo "Done! Fixing translation output..."
# grep the actual translation output
bash "$SCRIPTS/fix-trans-output.sh" \
    "${_arg_output_file}" \
    "${_arg_src}" \
    "${_arg_tgt}"

# for morsel & lmvr-tuned, stitch together behavior happens here
stitch_input="${_arg_output_file}.${_arg_tgt}"
stitch_output="${_arg_output_file}.stitched.${_arg_tgt}"
if [ "${_arg_model_type}" = "morsel" ] || [ "${_arg_model_type}" = "lmvr-tuned" ]; then
    echo "Stitching together MORSEL / LMVR-tuned segmented data..."
    python scripts/stitch-segmentations-together.py \
        --input-path "${stitch_input}" \
        --output-path "${stitch_output}" \
        --model-type "${_arg_model_type}"
else
    cp "${stitch_input}" "${stitch_output}"
fi

echo "Done! Detokenizing..."
# grep the actual translation output
detok_output="${_arg_output_file}.stitched.detok.${_arg_tgt}"
bash "$SCRIPTS/detokenize.sh" \
    "${stitch_output}" \
    "${detok_output}" \
    "${_arg_tgt}"

echo "Done! Validating number of lines..."
# validate that there are were no lines lost
lines_in_ref=$(wc -l "${_arg_reference}" | cut -d' ' -f1)
lines_in_output=$(wc -l "${_arg_output_file}.${_arg_tgt}" | cut -d' ' -f1)
lines_in_stitched=$(wc -l "${stitch_output}" | cut -d' ' -f1)
lines_in_detok=$(wc -l "${detok_output}" | cut -d' ' -f1)

for out in "${lines_in_output}" "${lines_in_detok}" "${lines_in_stitched}"; do
    if [ ! "${lines_in_ref}" = "${out}" ]; then
        echo "Line number mismatch!"
        echo "Reference: ${lines_in_ref}"
        echo "'out': ${out}"
        echo "Plain output: ${lines_in_output}"
        echo "Stitched: ${lines_in_stitched}"
        echo "Detokenized: ${lines_in_detok}"
        exit 1
    fi
done

echo "Done! Computing BLEU..."
# compute the BLEU score
bash "$SCRIPTS/score-with-sacrebleu.sh" \
    "${_arg_src}" "${_arg_tgt}" \
    "${detok_output}" \
    "${_arg_reference}" \
    "${_arg_output_file}.bleu.log"

echo "Done! Computing CHRF3..."
python "${SCRIPTS}/score-with-chrf.py" \
    --hypotheses-file "${detok_output}" \
    --references-file "${_arg_reference}" \
    --output-file "${_arg_output_file}.chrf3.log"

# ] <-- needed because of Argbash
