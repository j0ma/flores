# ultimate prepare ne-en

# constants
SRC=ne
TGT=en

BPESIZE=$1
if [ -z "$BPESIZE" ]; then
    BPESIZE=5000
fi

echo "BPE size = "$BPESIZE

if [ -z "${MOSES_SCRIPTS}" ]; then
    echo "Please make sure the MOSES_SCRIPTS environment variable is set!"
    exit 1
fi

MOSES_TOKENIZER_SCRIPT="$MOSES_SCRIPTS/tokenizer/tokenizer.perl"
MOSES_LOWERCASE_SCRIPT="$MOSES_SCRIPTS/tokenizer/lowercase.perl"
MOSES_CLEAN="$MOSES_SCRIPTS/training/clean-corpus-n.perl"
MOSES_NORM_PUNC="$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl"
MOSES_REM_NON_PRINT_CHAR="$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl"

original_preprocess() {

    # original preprocessing
    # with indic nlp etc.

    echo "pre-processing train data..."
    for FILE in "${TRAIN_SETS[@]}"; do
        $SRC_TOKENIZER $DATA/$FILE.$SRC
    done >$TMP/train.$SRC
    for FILE in "${TRAIN_SETS[@]}"; do
        $TGT_TOKENIZER $DATA/$FILE.$TGT
    done >$TMP/train.$TGT

    echo "pre-processing dev/test data..."
    $SRC_TOKENIZER $DATA/${VALID_SET}.$SRC >$TMP/valid.$SRC
    $TGT_TOKENIZER $DATA/${VALID_SET}.$TGT >$TMP/valid.$TGT
    $SRC_TOKENIZER $DATA/${TEST_SET}.$SRC >$TMP/test.$SRC
    $TGT_TOKENIZER $DATA/${TEST_SET}.$TGT >$TMP/test.$TGT
}

moses_pipeline() {

    # Pipeline for Moses tokenization
    # and other preprocessing functions.

    # NOTE: since indic_nlp_library is
    # used outside of this function, we
    # only use Moses on English data.

    INPUT_FILE=$1
    OUTPUT_FILE=$2
    LANGUAGE=$3

    if [ "$LANGUAGE" == "en" ]; then
        cat "$INPUT_FILE" |
            perl "$MOSES_NORM_PUNC $LANGUAGE" |
            perl "$MOSES_REM_NON_PRINT_CHAR" |
            perl "$MOSES_TOKENIZER_SCRIPT" \
                >"$OUTPUT_FILE"
    else
        cp "$INPUT_FILE" "$OUTPUT_FILE"
    fi
}

convert_lowercase() {

    INPUT_FILE=$1
    OUTPUT_FILE=$2
    if [ "$LANGUAGE" == "en" ]; then
        "$MOSES_LOWERCASE_SCRIPT" \
            <"$INPUT_FILE" >"$OUTPUT_FILE"
    else
        cp "$INPUT_FILE" "$OUTPUT_FILE"
    fi
}

TRAIN_MINLEN=1   # remove sentences with <1 BPE token
TRAIN_MAXLEN=250 # remove sentences with >250 BPE tokens

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=$ROOT/data

SRC_TOKENIZER="bash $SCRIPTS/indic_norm_tok.sh $SRC"
TGT_TOKENIZER="cat" # learn target-side BPE over untokenized (raw) text
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

# download if necessary
URLS=("https://github.com/facebookresearch/flores/raw/master/data/wikipedia_en_ne_si_test_sets.tgz"
)
ARCHIVES=(
    "wikipedia_en_ne_si_test_sets.tgz"
)
TRAIN_SETS=(
    "all-clean-ne/bible_dup.en-ne"
    "all-clean-ne/bible.en-ne"
    "all-clean-ne/globalvoices.2018q4.ne-en"
    "all-clean-ne/GNOMEKDEUbuntu.en-ne"
    "all-clean-ne/nepali-penn-treebank"
)
VALID_SET="wikipedia_en_ne_si_test_sets/wikipedia.dev.ne-en"
TEST_SET="wikipedia_en_ne_si_test_sets/wikipedia.devtest.ne-en"

if [ ! -d $DATA/all-clean-ne ]; then
    echo "Data directory not found. Please run 'bash download-data.sh' first..."
    exit -1
fi

# download and extract data
for ((i = 0; i < ${#URLS[@]}; ++i)); do
    ARCHIVE=$DATA/${ARCHIVES[i]}
    if [ -f $ARCHIVE ]; then
        echo "$ARCHIVE already exists, skipping download"
    else
        URL=${URLS[i]}
        wget -P $DATA "$URL"
        if [ -f $ARCHIVE ]; then
            echo "$URL successfully downloaded."
        else
            echo "$URL not successfully downloaded."
            exit -1
        fi
    fi
    FILE=${ARCHIVE: -4}
    if [ -e $FILE ]; then
        echo "$FILE already exists, skipping extraction"
    else
        tar -C $DATA -xzvf $ARCHIVE
    fi
done

#####################################
#   JOINT & NONJOINT SENTENCEPIECE  #
#   - these operate on raw text     #
#####################################

# sentencepiece joint
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}
mkdir -p $TMP $DATABIN

echo "Running joint sentencepiece..."
bash $SCRIPTS/download_indic.sh
original_preprocess

#for FILE in "${TRAIN_SETS[@]}"; do
    #$SRC_TOKENIZER $DATA/$FILE.$SRC
#done >$TMP/train.$SRC
#for FILE in "${TRAIN_SETS[@]}"; do
    #$TGT_TOKENIZER $DATA/$FILE.$TGT
#done >$TMP/train.$TGT

#echo "pre-processing dev/test data..."
#$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC >$TMP/valid.$SRC
#$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT >$TMP/valid.$TGT
#$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC >$TMP/test.$SRC
#$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT >$TMP/test.$TGT

# learn BPE with sentencepiece
python $SPM_TRAIN \
    --input=$TMP/train.$SRC,$TMP/train.$TGT \
    --model_prefix=$DATABIN/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode train/valid/test
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.bpe.model \
    --output_format=piece \
    --inputs $TMP/train.$SRC $TMP/train.$TGT \
    --outputs $TMP/train.bpe.$SRC $TMP/train.bpe.$TGT \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
for SPLIT in "valid" "test"; do
    python $SPM_ENCODE \
        --model $DATABIN/sentencepiece.bpe.model \
        --output_format=piece \
        --inputs $TMP/$SPLIT.$SRC $TMP/$SPLIT.$TGT \
        --outputs $TMP/$SPLIT.bpe.$SRC $TMP/$SPLIT.bpe.$TGT
done

# binarize data
fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TMP/train.bpe --validpref $TMP/valid.bpe --testpref $TMP/test.bpe \
    --destdir $DATABIN \
    --joined-dictionary \
    --workers 4

# sentencepiece nonjoint
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_nonjoint
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_lowercase
mkdir -p $TMP $DATABIN
echo "Running nonjoint sentencepiece..."
bash $SCRIPTS/download_indic.sh

for FILE in "${TRAIN_SETS[@]}"; do
    $SRC_TOKENIZER $DATA/$FILE.$SRC
done >$TMP/train.$SRC
for FILE in "${TRAIN_SETS[@]}"; do
    $TGT_TOKENIZER $DATA/$FILE.$TGT
done >$TMP/train.$TGT

echo "pre-processing dev/test data..."
$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC >$TMP/valid.$SRC
$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT >$TMP/valid.$TGT
$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC >$TMP/test.$SRC
$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT >$TMP/test.$TGT

# learn source side BPE with sentencepiece
python $SPM_TRAIN \
    --input=$TMP/train.$SRC \
    --model_prefix=$DATABIN/sentencepiece.$SRC.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# learn target side BPE with sentencepiece
python $SPM_TRAIN \
    --input=$TMP/train.$TGT \
    --model_prefix=$DATABIN/sentencepiece.$TGT.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

#--model $DATABIN/sentencepiece.$SRC.bpe.model \
# encode source & target side train/valid/test
python $SPM_ENCODE \
    --inputs $TMP/train.$SRC $TMP/train.$TGT \
    --outputs $TMP/train.bpe.$SRC $TMP/train.bpe.$TGT \
    --output_format=piece \
    --model_src $DATABIN/sentencepiece.$SRC.bpe.model \
    --model_tgt $DATABIN/sentencepiece.$TGT.bpe.model \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
for SPLIT in "valid" "test"; do
    python $SPM_ENCODE \
        --model_src $DATABIN/sentencepiece.$SRC.bpe.model \
        --model_tgt $DATABIN/sentencepiece.$TGT.bpe.model \
        --output_format=piece \
        --inputs $TMP/$SPLIT.$SRC $TMP/$SPLIT.$TGT \
        --outputs $TMP/$SPLIT.bpe.$SRC $TMP/$SPLIT.bpe.$TGT

done

# binarize data
fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TMP/train.bpe \
    --validpref $TMP/valid.bpe \
    --testpref $TMP/test.bpe \
    --destdir $DATABIN \
    --joined-dictionary \
    --workers 4

#######################################
#   JOINT SENTENCEPIECE W/LOWERCASING #
#   - lowercased input before BPE     #
#######################################

# vanilla + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_lowercase
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_lowercase
mkdir -p $TMP $DATABIN

echo "pre-processing train data..."
for FILE in "${TRAIN_SETS[@]}"; do
    $SRC_TOKENIZER $DATA/$FILE.$SRC
done >$TMP/train.$SRC
for FILE in "${TRAIN_SETS[@]}"; do
    $TGT_TOKENIZER $DATA/$FILE.$TGT
done >$TMP/train.$TGT

echo "pre-processing dev/test data..."
$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC >$TMP/valid.$SRC
$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT >$TMP/valid.$TGT
$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC >$TMP/test.$SRC
$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT >$TMP/test.$TGT

# lowercase english side
$SCRIPTS/lowercase.sh $TMP

# learn BPE with sentencepiece
python $SPM_TRAIN \
    --input=$TMP/train.$SRC,$TMP/train.$TGT \
    --model_prefix=$DATABIN/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode train/valid/test
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.bpe.model \
    --output_format=piece \
    --inputs $TMP/train.$SRC $TMP/train.$TGT \
    --outputs $TMP/train.bpe.$SRC $TMP/train.bpe.$TGT \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
for SPLIT in "valid" "test"; do
    python $SPM_ENCODE \
        --model $DATABIN/sentencepiece.bpe.model \
        --output_format=piece \
        --inputs $TMP/$SPLIT.$SRC $TMP/$SPLIT.$TGT \
        --outputs $TMP/$SPLIT.bpe.$SRC $TMP/$SPLIT.bpe.$TGT
done

# binarize data
fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TMP/train.bpe \
    --validpref $TMP/valid.bpe \
    --testpref $TMP/test.bpe \
    --destdir $DATABIN \
    --joined-dictionary \
    --workers 4

#############################################
#   MOSES TOKENIZATION + SUBWORD-NMT BPE    #
#############################################

# subword-nmt + moses + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_subwordnmt
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_subwordnmt
mkdir -p "$TMP" "$DATABIN"

echo "pre-processing train data..."
for FILE in "${TRAIN_SETS[@]}"; do
    $SRC_TOKENIZER $DATA/$FILE.$SRC
done >$TMP/train.$SRC
for FILE in "${TRAIN_SETS[@]}"; do
    $TGT_TOKENIZER $DATA/$FILE.$TGT
done >$TMP/train.$TGT

echo "pre-processing dev/test data..."
$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC >$TMP/valid.$SRC
$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT >$TMP/valid.$TGT
$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC >$TMP/test.$SRC
$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT >$TMP/test.$TGT

TMP_BIN=$ROOT/morfessor-models/
mkdir -p "$TMP_BIN"

for KIND in "train" "valid" "test"; do
    for LANGUAGE in ne en; do

        # note: in case LANGUAGE != "en",
        # only copying is performed

        moses_pipeline \
            "$TMP/$KIND.$LANGUAGE" \
            "$TMP/$KIND.$LANGUAGE.tok" \
            $LANGUAGE

        convert_lowercase \
            "$TMP/$KIND.$LANGUAGE.tok" \
            "$TMP/$KIND.$LANGUAGE.tok.lower"

        SEGM_INPUT_FILE=$TMP/$KIND.$LANGUAGE.tok.lower
        SEGM_OUTPUT_FILE=$TMP/$KIND.subword-nmt.$LANGUAGE
        bash "$SCRIPTS/segment.sh" \
            --input "$SEGM_INPUT_FILE" \
            --output "$SEGM_OUTPUT_FILE" \
            --model subword-nmt \
            --model-binary none \
            --bpe-size "$BPESIZE"
    done
done

for KIND in "train" "valid"; do
    for LANGUAGE in ne en; do
        perl "$MOSES_CLEAN" \
            -ratio 1.5 \
            "$TMP/$KIND.subword-nmt" \
            "$SRC" "$TGT" \
            "$TMP/$KIND.subword-nmt.clean" \
            "$TRAIN_MINLEN" \
            "$TRAIN_MAXLEN"
    done
done

# binarize data
fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TMP/train.subword-nmt.clean \
    --validpref $TMP/valid.subword-nmt.clean \
    --testpref $TMP/test.subword-nmt.clean \
    --destdir $DATABIN \
    --joined-dictionary \
    --workers 4

###############################################
#   MOSES TOKENIZATION + MORFESSOR BASELINE   #
###############################################

# morfessor baseline + moses + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_morfessorbaseline
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_morfessorbaseline
mkdir -p $TMP $DATABIN

echo "pre-processing train data..."
for FILE in "${TRAIN_SETS[@]}"; do
    $SRC_TOKENIZER $DATA/$FILE.$SRC
done >$TMP/train.$SRC
for FILE in "${TRAIN_SETS[@]}"; do
    $TGT_TOKENIZER $DATA/$FILE.$TGT
done >$TMP/train.$TGT

echo "pre-processing dev/test data..."
$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC >$TMP/valid.$SRC
$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT >$TMP/valid.$TGT
$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC >$TMP/test.$SRC
$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT >$TMP/test.$TGT

## use pre-trained morfessor models
TMP_BIN=$ROOT/morfessor-models/
mkdir -p $TMP_BIN

for KIND in "train" "valid" "test"; do
    for LANGUAGE in ne en; do

        moses_pipeline \
            "$TMP/$KIND.$LANGUAGE" \
            "$TMP/$KIND.$LANGUAGE.tok"

        convert_lowercase \
            "$TMP/$KIND.$LANGUAGE.tok" \
            "$TMP/$KIND.$LANGUAGE.tok.lower"

        MF_SEGM_INPUT_FILE=$TMP/$KIND.$LANGUAGE.tok.lower
        MF_SEGM_OUTPUT_FILE=$TMP/$KIND.morfessor-baseline.$LANGUAGE
        MF_SEGM_MODEL_FILE=$TMP_BIN/flores.vocab.$LANGUAGE.lowercase-morfessor-baseline-batch-recursive-$LANGUAGE.bin
        bash "$SCRIPTS/segment.sh" \
            --input "$MF_SEGM_INPUT_FILE" \
            --output "$MF_SEGM_OUTPUT_FILE" \
            --model baseline \
            --model-binary "$MF_SEGM_MODEL_FILE"

    done
done

for KIND in "train" "valid"; do
    for LANGUAGE in ne en; do
        perl "$MOSES_CLEAN" \
            -ratio 1.5 \
            "$TMP/$KIND.morfessor-baseline" \
            "$SRC" "$TGT" \
            "$TMP/$KIND.morfessor-baseline.clean" \
            "$TRAIN_MINLEN" \
            "$TRAIN_MAXLEN"
    done
done

# binarize data
fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TMP/train.morfessor-baseline.clean \
    --validpref $TMP/valid.morfessor-baseline.clean \
    --testpref $TMP/test.morfessor-baseline.clean \
    --destdir $DATABIN \
    --joined-dictionary \
    --workers 4

###############################################
#   MOSES TOKENIZATION + MORFESSOR FLATCAT    #
###############################################

# morfessor flatcat + moses + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_flatcat
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_flatcat
mkdir -p $TMP $DATABIN

echo "pre-processing train data..."
for FILE in "${TRAIN_SETS[@]}"; do
    $SRC_TOKENIZER $DATA/$FILE.$SRC
done >$TMP/train.$SRC
for FILE in "${TRAIN_SETS[@]}"; do
    $TGT_TOKENIZER $DATA/$FILE.$TGT
done >$TMP/train.$TGT

echo "pre-processing dev/test data..."
$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC >$TMP/valid.$SRC
$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT >$TMP/valid.$TGT
$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC >$TMP/test.$SRC
$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT >$TMP/test.$TGT

## use pre-trained morfessor models
TMP_BIN=$ROOT/morfessor-models/
mkdir -p $TMP_BIN

for KIND in "train" "valid" "test"; do
    for LANGUAGE in ne en; do

        moses_pipeline \
            "$TMP/$KIND.$LANGUAGE" \
            "$TMP/$KIND.$LANGUAGE.tok"

        convert_lowercase \
            "$TMP/$KIND.$LANGUAGE.tok" \
            "$TMP/$KIND.$LANGUAGE.tok.lower"

        MF_SEGM_INPUT_FILE=$TMP/$KIND.$LANGUAGE.lower.tok
        MF_SEGM_OUTPUT_FILE=$TMP/$KIND.morfessor-baseline.$LANGUAGE
        MF_SEGM_MODEL_FILE=$TMP_BIN/flores.vocab.$LANGUAGE.lowercase-morfessor-flatcat-batch-$LANGUAGE.bin
        bash "$SCRIPTS/segment.sh" \
            --input "$MF_SEGM_INPUT_FILE" \
            --output "$MF_SEGM_OUTPUT_FILE" \
            --model flatcat \
            --model-binary "$MF_SEGM_MODEL_FILE"

    done
done

for KIND in "train" "valid"; do
    for LANGUAGE in ne en; do
        perl "$MOSES_CLEAN" \
            -ratio 1.5 \
            "$TMP/$KIND.morfessor-flatcat" \
            "$SRC" "$TGT" \
            "$TMP/$KIND.morfessor-flatcat.clean" \
            "$TRAIN_MINLEN" \
            "$TRAIN_MAXLEN"
    done
done

fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TMP/train.morfessor-flatcat.clean \
    --validpref $TMP/valid.morfessor-flatcat.clean \
    --testpref $TMP/test.morfessor-flatcat.clean \
    --destdir $DATABIN \
    --joined-dictionary \
    --workers 4

#################################################
#   MOSES TOKENIZATION + LMVR (Ataman, 2017)    #
#################################################

# lmvr + moses + lowercase
