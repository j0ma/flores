#!/bin/bash

# Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([checkpoint-glob])
# ARG_OPTIONAL_SINGLE([src])
# ARG_OPTIONAL_SINGLE([tgt])
# ARG_OPTIONAL_SINGLE([eval-on])
# ARG_OPTIONAL_SINGLE([data-folder])
# ARG_OPTIONAL_SINGLE([data-bin-folder])
# ARG_OPTIONAL_SINGLE([segmentation-model-type])
# ARG_OPTIONAL_SINGLE([model-name])
# ARG_OPTIONAL_SINGLE([reference])
# ARG_OPTIONAL_SINGLE([remove-bpe-type])
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
_arg_checkpoint_glob=
_arg_src=
_arg_tgt=
_arg_eval_on=
_arg_data_folder=
_arg_data_bin_folder=
_arg_segmentation_model_type=
_arg_model_name=
_arg_reference=
_arg_remove_bpe_type=

print_help() {
    printf '%s\n' "<The general help message of my script>"
    printf 'Usage: %s [--checkpoint-glob <arg>] [--src <arg>] [--tgt <arg>] [--eval-on <arg>] [--data-folder <arg>] [--data-bin-folder <arg>] [--segmentation-model-type <arg>] [--model-name <arg>] [--reference <arg>] [--remove-bpe-type <arg>] [-h|--help]\n' "$0"
    printf '\t%s\n' "-h, --help: Prints help"
}

parse_commandline() {
    while test $# -gt 0; do
        _key="$1"
        case "$_key" in
        --checkpoint-glob)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_checkpoint_glob="$2"
            shift
            ;;
        --checkpoint-glob=*)
            _arg_checkpoint_glob="${_key##--checkpoint-glob=}"
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
        --segmentation-model-type)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_segmentation_model_type="$2"
            shift
            ;;
        --segmentation-model-type=*)
            _arg_segmentation_model_type="${_key##--segmentation-model-type=}"
            ;;
        --model-name)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_model_name="$2"
            shift
            ;;
        --model-name=*)
            _arg_model_name="${_key##--model-name=}"
            ;;
        --reference)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_reference="$2"
            shift
            ;;
        --reference=*)
            _arg_reference="${_key##--reference=}"
            ;;
        --remove-bpe-type)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_remove_bpe_type="$2"
            shift
            ;;
        --remove-bpe-type=*)
            _arg_remove_bpe_type="${_key##--remove-bpe-type=}"
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

printf 'Value of --%s: %s\n' 'checkpoint-glob' "$_arg_checkpoint_glob"
printf 'Value of --%s: %s\n' 'src' "$_arg_src"
printf 'Value of --%s: %s\n' 'tgt' "$_arg_tgt"
printf 'Value of --%s: %s\n' 'eval-on' "$_arg_eval_on"
printf 'Value of --%s: %s\n' 'data-folder' "$_arg_data_folder"
printf 'Value of --%s: %s\n' 'data-bin-folder' "$_arg_data_bin_folder"
printf 'Value of --%s: %s\n' 'segmentation-model-type' "$_arg_segmentation_model_type"
printf 'Value of --%s: %s\n' 'model-name' "$_arg_model_name"
printf 'Value of --%s: %s\n' 'reference' "$_arg_reference"
printf 'Value of --%s: %s\n' 'remove-bpe-type' "$_arg_remove_bpe_type"

get_seed() {
    echo "$1" |
        grep -Eo "seed[0-9]{2}" |
        sed "s/seed//g"
}

for checkpoint in $(ls "${_arg_checkpoint_glob}"); do
    _seed=$(get_seed "${checkpoint}")
    echo "Seed: ${_seed}"
    [ -z "${_seed}" ] && echo "Seed not found!" && exit 1
    bash eval-model.sh \
        --src "${_arg_src}" --tgt "${_arg_tgt}" \
        --eval-on "${_arg_eval_on}" \
        --data-folder "${_arg_data_folder}" \
        --data-bin-folder "${_arg_data_bin_folder}" \
        --model-checkpoint "${checkpoint}" \
        --model-type "${_arg_segmentation_model_type}" \
        --output-file "./translation-output-wmt19/${_arg_model_name}/seed-${_seed}/en-ne.output.raw" \
        --reference "${_arg_reference}" \
        --remove-bpe "${_arg_remove_bpe_type}" &
done
wait

# ] <-- needed because of Argbash
