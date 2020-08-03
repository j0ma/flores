#!/bin/bash

# Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([src])
# ARG_OPTIONAL_SINGLE([tgt])
# ARG_OPTIONAL_SINGLE([eval-on])
# ARG_OPTIONAL_SINGLE([input-path])
# ARG_OPTIONAL_SINGLE([data-folder])
# ARG_OPTIONAL_SINGLE([data-bin-folder])
# ARG_OPTIONAL_SINGLE([model-checkpoint])
# ARG_OPTIONAL_SINGLE([model-type])
# ARG_OPTIONAL_SINGLE([output-file])
# ARG_OPTIONAL_SINGLE([remove-bpe])

# ARG_HELP([<The general help message of my script>])
# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.8.1 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info

set -eo pipefail

die()
{
	local _ret=$2
	test -n "$_ret" || _ret=1
	test "$_PRINT_HELP" = yes && print_help >&2
	echo "$1" >&2
	exit ${_ret}
}


begins_with_short_option()
{
	local first_option all_short_options='h'
	first_option="${1:0:1}"
	test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_src=
_arg_tgt=
_arg_eval_on=
_arg_input_path=
_arg_data_folder=
_arg_model_checkpoint=
_arg_model_type=
_arg_output_file=
_arg_remove_bpe="regular"


print_help()
{
	printf '%s\n' "Evaluate your NMT model & produce actual output!"
	printf 'Usage: %s [--src <arg>] [--tgt <arg>] [--eval-on <arg>] [--input-folder <arg>]  [--data-folder <arg>] [--data-bin-folder <arg>] [--model-checkpoint <arg>] [--model-type <arg>] [--output-file <arg>] [-h|--help]\n' "$0"
	printf '\t%s\n' "-h, --help: Prints help"
}


parse_commandline()
{
	while test $# -gt 0
	do
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
			--input-path)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_input_path="$2"
				shift
				;;
			--input-path=*)
				_arg_input_path="${_key##--input-path=}"
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
			--remove-bpe)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_remove_bpe="$2"
				shift
				;;
			--remove-bpe=*)
				_arg_remove_bpe="${_key##--remove-bpe=}"
				;;
			-h|--help)
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

evaluate_fairseq_interactive () {
    SRC_LANG=$1
    TGT_LANG=$2
    CHECKPOINT_PATH=$3
    DATA_DIR=$4
    REMOVE_BPE=$5
    SPLIT=$6

    FAIRSEQ_CMD="""\
        fairseq-interactive \
            "${DATA_DIR}" \
            --source-lang "${SRC_LANG}" \
            --target-lang "${TGT_LANG}" \
            --path "${CHECKPOINT_PATH}" \
            --beam 5 --lenpen 1.2 --cpu \
            --gen-subset "${SPLIT}" \
            --max-sentences 256 \
            --buffer-size 256 \
    """

    case  "${REMOVE_BPE}"  in
        "regular"|"standard")       
            FAIRSEQ_CMD="${FAIRSEQ_CMD} --remove-bpe"
            ;;
        "sentencepiece")       
            FAIRSEQ_CMD="${FAIRSEQ_CMD} --remove-bpe=sentencepiece"
            ;;
        *)              
            echo "Not removing BPE!"
    esac 

    #if [ "${REMOVE_BPE}" = "regular" ] || [ "${REMOVE_BPE}" = "standard" ]
    #then
        #FAIRSEQ_CMD="${FAIRSEQ_CMD} --remove-bpe"
    #elif [ "${REMOVE_BPE}" = "sentencepiece" ]
    #then
        #FAIRSEQ_CMD="${FAIRSEQ_CMD} --remove-bpe=sentencepiece"
    #else
        #echo "Not removing BPE!"
    #fi

    if [ ! "${SRC_LANG}" = "en" ];
    then
        FAIRSEQ_CMD="${FAIRSEQ_CMD} --sacrebleu"
    fi

    echo "Invoking fairseq-interactive"
    $FAIRSEQ_CMD
}

printf 'Value of --%s: %s\n' 'src' "$_arg_src"
printf 'Value of --%s: %s\n' 'tgt' "$_arg_tgt"
printf 'Value of --%s: %s\n' 'eval-on' "$_arg_eval_on"
printf 'Value of --%s: %s\n' 'data-folder' "$_arg_data_folder"
printf 'Value of --%s: %s\n' 'data-bin-folder' "$_arg_data_bin_folder"
printf 'Value of --%s: %s\n' 'model-checkpoint' "$_arg_model_checkpoint"
printf 'Value of --%s: %s\n' 'model-type' "$_arg_model_type"
printf 'Value of --%s: %s\n' 'output-file' "$_arg_output_file"
printf 'Value of --%s: %s\n' 'remove-bpe' "$_arg_remove_bpe"

if [ -z "${_arg_input_path}" ]; then
    INPUT_PATH="${_arg_input_path}"
else
    INPUT_PATH="${_arg_data_folder}/${_arg_eval_on}.${_arg_model_type}.${_arg_src}"
fi

cat "${INPUT_PATH}" |
    evaluate_fairseq_interactive \
        "${_arg_src}" \
        "${_arg_tgt}" \
        "${_arg_model_checkpoint}" \
        "${_arg_data_bin_folder}" \
        "${_arg_remove_bpe}" \
        "${_arg_eval_on}" \
            > "${_arg_output_file}"

# ] <-- needed because of Argbash
