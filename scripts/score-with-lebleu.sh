# score-with-lebleu.sh

PRED=$1
REF=$2
LOGFILE=$3

# make sure we're actually running 2.7
if [ -z "$(python -c "import sys; print(sys.version)" | grep -E "^2\.7")" ]; then
    echo "Need to be running Python 2.7 for LMVR!"
    exit 1
fi

# make sure lebleu executable exists
[[ -z "${LEBLEU_PATH}" ]] && \
    echo "LEBLEU_PATH variable must be set!" \
    && exit 1

LEBLEU="${LEBLEU_PATH}/cmd.py"
python "${LEBLEU}" "${PRED}" "${REF}" > "${LOGFILE}"
