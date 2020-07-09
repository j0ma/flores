SRC=$1
TGT=$2
PRED=$3
REF=$4

if [ "${TGT}"="en" ]; then
    TOK="13a"
else
    TOK="none"
fi

cat "${PRED}" |
    sacrebleu \
        --language-pair "${SRC}-${TGT}" \
        -lc --tokenize "${TOK}" \
        "${REF}"



