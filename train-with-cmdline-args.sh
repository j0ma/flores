#!/bin/bash

# Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([src])
# ARG_OPTIONAL_SINGLE([tgt])
# ARG_OPTIONAL_SINGLE([slug])
# ARG_OPTIONAL_SINGLE([segmentation-method])
# ARG_OPTIONAL_SINGLE([bpe-size])
# ARG_OPTIONAL_SINGLE([cuda-device])
# ARG_OPTIONAL_SINGLE([clip-norm])
# ARG_OPTIONAL_SINGLE([from-seed])
# ARG_OPTIONAL_SINGLE([to-seed])
# ARG_OPTIONAL_BOOLEAN([fp16])
# ARG_HELP([Script for training Flores using command line arguments])
# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.8.1 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info

set -euo pipefail

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
#_arg_slug=
#_arg_segmentation_method=
_arg_bpe_size=5000
#_arg_cuda_device=
_arg_clip_norm=0.1
_arg_from_seed=10
_arg_to_seed=10
_arg_fp16="on"

print_help() {
    printf '%s\n' "Script for training Flores using command line arguments"
    printf 'Usage: %s [--src <arg>] [--tgt <arg>] [--slug <arg>] [--segmentation-method <arg>] [--bpe-size <arg>] [--cuda-device <arg>] [--clip-norm <arg>] [--from-seed <arg>] [--to-seed <arg>] [--(no-)fp16] [-h|--help]\n' "$0"
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
        --slug)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_slug="$2"
            shift
            ;;
        --slug=*)
            _arg_slug="${_key##--slug=}"
            ;;
        --segmentation-method)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_segmentation_method="$2"
            shift
            ;;
        --segmentation-method=*)
            _arg_segmentation_method="${_key##--segmentation-method=}"
            ;;
        --bpe-size)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_bpe_size="$2"
            shift
            ;;
        --bpe-size=*)
            _arg_bpe_size="${_key##--bpe-size=}"
            ;;
        --cuda-device)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_cuda_device="$2"
            shift
            ;;
        --cuda-device=*)
            _arg_cuda_device="${_key##--cuda-device=}"
            ;;
        --clip-norm)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_clip_norm="$2"
            shift
            ;;
        --clip-norm=*)
            _arg_clip_norm="${_key##--clip-norm=}"
            ;;
        --from-seed)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_from_seed="$2"
            shift
            ;;
        --from-seed=*)
            _arg_from_seed="${_key##--from-seed=}"
            ;;
        --to-seed)
            test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
            _arg_to_seed="$2"
            shift
            ;;
        --to-seed=*)
            _arg_to_seed="${_key##--to-seed=}"
            ;;
        --no-fp16 | --fp16)
            _arg_fp16="on"
            test "${1:0:5}" = "--no-" && _arg_fp16="off"
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

# Define custom functions

train_fairseq() {

    # fairseq training with custom clip_norm

    SRC_LANG=$1
    TGT_LANG=$2
    CHECKPOINT_DIR=$3
    DATA_DIR=$4
    RAND_SEED=$5
    CUDA_DEVICE=$6
    CLIP_NORM=$7

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE fairseq-train \
        "$DATA_DIR" \
        --source-lang "$SRC_LANG" --target-lang "$TGT_LANG" \
        --arch transformer --share-all-embeddings \
        --encoder-layers 5 --decoder-layers 5 \
        --encoder-embed-dim 512 --decoder-embed-dim 512 \
        --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
        --encoder-attention-heads 2 --decoder-attention-heads 2 \
        --encoder-normalize-before --decoder-normalize-before \
        --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
        --weight-decay 0.0001 \
        --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --clip-norm "${CLIP_NORM}" \
        --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
        --lr 1e-3 --min-lr 1e-9 \
        --max-tokens 4000 \
        --update-freq 4 \
        --max-epoch 100 \
        --save-interval 10 \
        --save-dir "$CHECKPOINT_DIR" \
        --seed "$RAND_SEED" \
        --fp16
}

train() {

    SRC_LANG=$1
    TGT_LANG=$2
    SRC_LANG_CAP=$(echo "$SRC_LANG" | awk '{print toupper($0)}')
    TGT_LANG_CAP=$(echo "$TGT_LANG" | awk '{print toupper($0)}')
    RAND_SEED=$3
    SEGM_METHOD=$4
    BPE_SIZE=$5
    CUDA_DEVICE=$6
    CLIP_NORM=$7
    LOG_DIR=$8
    CHECKPOINT_DIR=$9

    if [ -z "$BPE_SIZE" ]; then
        BPE_SIZE=5000
    fi

    # create path for log file
    LOG_FILE="baseline_${SRC_LANG}_${TGT_LANG}.log"
    LOG_OUTPUT_PATH="$LOG_DIR/$LOG_FILE"

    echo "================ FLORES BASELINE WITH CLIP_NORM=${CLIP_NORM}, SEGMENTATION=${SEGM_METHOD}, BPE=${BPE_SIZE} AND SEED=${RAND_SEED} ================" >>"$LOG_OUTPUT_PATH"
    echo "About to train the supervised for the following language pair: ${SRC_LANG_CAP}-${TGT_LANG_CAP}" >>"$LOG_OUTPUT_PATH"
    echo "Logging output to: $LOG_OUTPUT_PATH"

    # create path to checkpoint directory
    echo "CHECKPOINT_DIR is set to ${CHECKPOINT_DIR}" >>"$LOG_OUTPUT_PATH"
    echo "Creating checkpoint directory if it doesn't exist..." >">$LOG_OUTPUT_PATH"
    mkdir -p "$CHECKPOINT_DIR"

    # create data directory
    if [ "$SRC_LANG" = "si" ] || [ "$TGT_LANG" = "si" ]; then
        DATA_DIR_PREFIX="data-bin/wiki_si_en"
    else
        DATA_DIR_PREFIX="data-bin/wiki_ne_en"
    fi

    case "$SEGM_METHOD" in
    sentencepiece-joint)
        DATA_DIR="${DATA_DIR_PREFIX}_bpe${BPE_SIZE}_joint"
        ;;
    sentencepiece-nonjoint)
        DATA_DIR="${DATA_DIR_PREFIX}_bpe${BPE_SIZE}_nonjoint"
        ;;
    sentencepiece-lowercase)
        DATA_DIR="${DATA_DIR_PREFIX}_bpe${BPE_SIZE}_lowercase"
        ;;
    subword-nmt)
        DATA_DIR="${DATA_DIR_PREFIX}_bpe${BPE_SIZE}_subwordnmt"
        ;;
    morfessor-baseline)
        DATA_DIR="${DATA_DIR_PREFIX}_morfessorbaseline"
        ;;
    morfessor-flatcat | flatcat)
        DATA_DIR="${DATA_DIR_PREFIX}_flatcat"
        ;;
    lmvr)
        DATA_DIR="${DATA_DIR_PREFIX}_lmvr"
        ;;
    morsel)
        DATA_DIR="${DATA_DIR_PREFIX}_morsel"
        ;;
    *)
        die "FATAL ERROR: Got an unexpected segmentation method! '$1'" 1
        ;;
    esac

    echo "Data folder is: ${DATA_DIR}" >>"$LOG_OUTPUT_PATH"

    # actually run the training script and pass in necessary env variable
    echo "Beginning training.:.." >>"$LOG_OUTPUT_PATH"
    echo "Time at beginning: $(date)" >>"$LOG_OUTPUT_PATH"
    train_fairseq \
        "${SRC_LANG}" \
        "${TGT_LANG}" \
        "${CHECKPOINT_DIR}" \
        "${DATA_DIR}" \
        "${RAND_SEED}" \
        "${CUDA_DEVICE}" \
        "${CLIP_NORM}" >>"${LOG_OUTPUT_PATH}"
    echo "Done training." >>$"${LOG_OUTPUT_PATH}"
    echo "Time at end: $(date)" >>"${LOG_OUTPUT_PATH}"
}

# End of custom functions

printf 'Value of --%s: %s\n' 'src' "$_arg_src"
printf 'Value of --%s: %s\n' 'tgt' "$_arg_tgt"
printf 'Value of --%s: %s\n' 'slug' "$_arg_slug"
printf 'Value of --%s: %s\n' 'segmentation-method' "$_arg_segmentation_method"
printf 'Value of --%s: %s\n' 'bpe-size' "$_arg_bpe_size"
printf 'Value of --%s: %s\n' 'cuda-device' "$_arg_cuda_device"
printf 'Value of --%s: %s\n' 'clip-norm' "$_arg_clip_norm"
printf 'Value of --%s: %s\n' 'from-seed' "$_arg_from_seed"
printf 'Value of --%s: %s\n' 'to-seed' "$_arg_to_seed"
printf "'%s' is %s\\n" 'fp16' "$_arg_fp16"

for SEED in $(seq "$_arg_from_seed" "$_arg_to_seed"); do

    SLUG="$_arg_slug"-seed"$SEED"

    # 0. create log & checkpoint folder
    bash ./create_log_folder.sh "$SLUG"
    bash ./create_checkpoint_folder.sh "$SLUG"

    LOG_DIR=$(bash ./create_log_folder.sh "$SLUG")
    CHECKPOINT_DIR=$(bash ./create_checkpoint_folder.sh "$SLUG")

    train \
        "$_arg_src" \
        "$_arg_tgt" \
        "$SEED" \
        "$_arg_segmentation_method" \
        "$_arg_bpe_size" \
        "$_arg_cuda_device" \
        "$_arg_clip_norm" \
        "$LOG_DIR" \
        "$CHECKPOINT_DIR"

    RESULTS_DIR=$(bash ./create_results_folder.sh "$SLUG")
    bash ./evaluate-with-cmdline-args.sh \
        --src "$_arg_src" \
        --tgt "$_arg_tgt" \
        --segmentation-method "$_arg_segmentation_method" \
        --bpe-size "$_arg_bpe_size" \
        --cuda-device "$_arg_cuda_device" \
        --results-folder "$RESULTS_DIR" \
        --checkpoint-folder "$CHECKPOINT_DIR"

done

# ] <-- needed because of Argbash
