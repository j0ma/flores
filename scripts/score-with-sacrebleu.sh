SRC=$1
TGT=$2
PRED=$3
REF=$4
LOGFILE=$5

if [ "${TGT}" = "en" ]; then
    echo "Setting tokenizer = 13a"
    TOK="13a"
else
    echo "Setting tokenizer = none"
    TOK="none"
fi

cat "${PRED}" |
    sacrebleu \
        --language-pair "${SRC}-${TGT}" \
        -lc --tokenize "${TOK}" \
        "${REF}" \
        > "${LOGFILE}"
