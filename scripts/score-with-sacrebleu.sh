SRC=$1
TGT=$2
PRED=$3
REF=$4
LOGFILE=$5
MODE=$6

if [ "${MODE}" = "wmt19" ]; then
    cat "${PRED}" |
        sacrebleu \
            --test-set wmt19 \
            -lc \
            --language-pair "${SRC}-${TGT}" \
            > "${LOGFILE}"
else

    # otherwise default to flores

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
fi
