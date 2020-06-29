#!/bin/bash
# prepare-sien-all-segmentations.sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: does ALL preprocessing necessary
#   - sentencepiece joint / non-joint
#   - sentencepiece lowercase + joint (vanilla model)
#   - subword-nmt
#   - morfessor baseline
#   - morfessor flatcat
#   - ataman lmvr
#   - morsel

# constants
SRC=si
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

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=$ROOT/data

MOSES_TOKENIZER_SCRIPT="$MOSES_SCRIPTS/tokenizer/tokenizer.perl"
MOSES_DETOKENIZER_SCRIPT="$MOSES_SCRIPTS/tokenizer/detokenizer.perl"
MOSES_LOWERCASE_SCRIPT="$MOSES_SCRIPTS/tokenizer/lowercase.perl"
MOSES_CLEAN="$MOSES_SCRIPTS/training/clean-corpus-n.perl"
MOSES_NORM_PUNC="$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl"
MOSES_REM_NON_PRINT_CHAR="$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl"
UNESCAPE_HTML_SCRIPT="${SCRIPTS}/unescape_html.py"

original_preprocessing_loop() {

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
            sed "s/--/ -- /g" |
            perl "$MOSES_NORM_PUNC" "$LANGUAGE" |
            perl "$MOSES_REM_NON_PRINT_CHAR" |
            perl "$MOSES_TOKENIZER_SCRIPT" |
            perl -C -MHTML::Entities -pe 'decode_entities($_);' \
                >"$OUTPUT_FILE"
    else
        cat "$INPUT_FILE" |
            sed "s/--/ -- /g" \
                >"$OUTPUT_FILE"
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

TRAIN_MINLEN=6   # remove sentences with <6 BPE tokens
TRAIN_MAXLEN=250 # remove sentences with >250 BPE tokens

SRC_TOKENIZER="bash $SCRIPTS/indic_norm_tok.sh $SRC" TGT_TOKENIZER="cat" # learn target-side BPE over untokenized (raw) text
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

URLS=(
    "https://github.com/facebookresearch/flores/raw/master/data/wikipedia_en_ne_si_test_sets.tgz"
)
ARCHIVES=(
    "wikipedia_en_ne_si_test_sets.tgz"
)
TRAIN_SETS=(
    "all-clean-si/GNOMEKDEUbuntu.en-si"
    "all-clean-si/OpenSubtitles2018.en-si"
)
VALID_SET="wikipedia_en_ne_si_test_sets/wikipedia.dev.si-en"
TEST_SET="wikipedia_en_ne_si_test_sets/wikipedia.devtest.si-en"

if [ ! -d $DATA/all-clean-si ]; then
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

bash $SCRIPTS/download_indic.sh

######################################
##   JOINT & NONJOINT SENTENCEPIECE  #
##   - these operate on raw text     #
######################################

echo "#####################################"
echo "#   JOINT & NONJOINT SENTENCEPIECE  #"
echo "#####################################"

echo "Joint Sentencepiece..."

# sentencepiece joint
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_joint
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_joint
mkdir -p "$TMP" "$DATABIN"

echo "Running joint sentencepiece..."

original_preprocessing_loop

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

#######################

echo "Nonjoint Sentencepiece..."
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode_nonjoint.py

# sentencepiece nonjoint
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_nonjoint
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_nonjoint
mkdir -p "$TMP" "$DATABIN"

original_preprocessing_loop

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

echo "#######################################"
echo "#   JOINT SENTENCEPIECE + LOWERCASE   #"
echo "#######################################"

echo "Joint Sentencepiece + lowercasing with AWK (old)..."
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

# vanilla + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_lowercase
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_lowercase
mkdir -p "$TMP" "$DATABIN"

original_preprocessing_loop

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

###############################################
#   MOSES TOKENIZATION + MORFESSOR FLATCAT    #
###############################################

echo "###############################################"
echo "#   MOSES TOKENIZATION + MORFESSOR FLATCAT    #"
echo "###############################################"

# morfessor flatcat + moses + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_flatcat
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_flatcat
mkdir -p "$TMP" "$DATABIN"

original_preprocessing_loop

## use pre-trained morfessor models
TMP_BIN=$ROOT/segmentation-models/
mkdir -p $TMP_BIN

for KIND in "train" "valid" "test"; do
    for LANGUAGE in si en; do

        moses_pipeline \
            "$TMP/$KIND.$LANGUAGE" \
            "$TMP/$KIND.$LANGUAGE.tok" \
            "$LANGUAGE"

        convert_lowercase \
            "$TMP/$KIND.$LANGUAGE.tok" \
            "$TMP/$KIND.$LANGUAGE.tok.lower"

        MF_SEGM_INPUT_FILE=$TMP/$KIND.$LANGUAGE.tok.lower
        MF_SEGM_OUTPUT_FILE=$TMP/$KIND.morfessor-flatcat.$LANGUAGE
        MF_SEGM_MODEL_FILE=$TMP_BIN/flores.vocab.$LANGUAGE.lowercase-morfessor-flatcat-batch-$LANGUAGE.bin
        bash "$SCRIPTS/segment.sh" \
            --input "$MF_SEGM_INPUT_FILE" \
            --output "$MF_SEGM_OUTPUT_FILE" \
            --model flatcat \
            --model-binary "$MF_SEGM_MODEL_FILE"

    done
done

# comment out due to excessive pruning
#for LANGUAGE in si en; do
#perl "$MOSES_CLEAN" \
#-ratio 1.5 \
#"$TMP/train.morfessor-flatcat" \
#"$SRC" "$TGT" \
#"$TMP/train.morfessor-flatcat.clean" \
#"$TRAIN_MINLEN" \
#"$TRAIN_MAXLEN"
#done

# we don't filter valid or test

for LANGUAGE in si en; do
    # we don't filter valid or test
    cp $TMP/valid.morfessor-flatcat.$LANGUAGE $TMP/valid.morfessor-flatcat.clean.$LANGUAGE
    cp $TMP/test.morfessor-flatcat.$LANGUAGE $TMP/test.morfessor-flatcat.clean.$LANGUAGE
done

fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TMP/train.morfessor-flatcat.clean \
    --validpref $TMP/valid.morfessor-flatcat.clean \
    --testpref $TMP/test.morfessor-flatcat.clean \
    --destdir $DATABIN \
    --joined-dictionary \
    --workers 4

###############################################
#   MOSES TOKENIZATION + MORFESSOR BASELINE   #
###############################################

echo "###############################################"
echo "#   MOSES TOKENIZATION + MORFESSOR BASELINE   #"
echo "###############################################"

# morfessor baseline + moses + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_morfessorbaseline
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_morfessorbaseline
mkdir -p "$TMP" "$DATABIN"

original_preprocessing_loop

## use pre-trained morfessor models
TMP_BIN=$ROOT/segmentation-models/
mkdir -p $TMP_BIN

for KIND in "train" "valid" "test"; do
    for LANGUAGE in si en; do

        moses_pipeline \
            "$TMP/$KIND.$LANGUAGE" \
            "$TMP/$KIND.$LANGUAGE.tok" \
            "$LANGUAGE"

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

# comment out due to excessive pruning
#for LANGUAGE in si en; do
#perl "$MOSES_CLEAN" \
#-ratio 1.5 \
#"$TMP/train.morfessor-baseline" \
#"$SRC" "$TGT" \
#"$TMP/train.morfessor-baseline.clean" \
#"$TRAIN_MINLEN" \
#"$TRAIN_MAXLEN"
#done

for LANGUAGE in si en; do
    # we don't filter valid or test
    cp $TMP/valid.morfessor-baseline.$LANGUAGE $TMP/valid.morfessor-baseline.clean.$LANGUAGE
    cp $TMP/test.morfessor-baseline.$LANGUAGE $TMP/test.morfessor-baseline.clean.$LANGUAGE
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

############################################
#   MOSES TOKENIZATION + SUBWORD-NMT BPE   #
############################################

echo "#############################################"
echo "#   MOSES TOKENIZATION + SUBWORD-NMT BPE    #"
echo "#############################################"

# subword-nmt + moses + lowercase
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_subwordnmt
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_subwordnmt
mkdir -p "$TMP" "$DATABIN"

original_preprocessing_loop

TMP_BIN=$ROOT/segmentation-models/
mkdir -p "$TMP_BIN"
for KIND in "train" "valid" "test"; do
    for LANGUAGE in si en; do

        # note: in case LANGUAGE != "en",
        # only copying is performed

        moses_pipeline \
            "$TMP/$KIND.$LANGUAGE" \
            "$TMP/$KIND.$LANGUAGE.tok" \
            "$LANGUAGE"

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

# comment out due to excessive pruning
#for LANGUAGE in si en; do
#perl "$MOSES_CLEAN" \
#-ratio 1.5 \
#"$TMP/train.subword-nmt" \
#"$SRC" "$TGT" \
#"$TMP/train.subword-nmt.clean" \
#"$TRAIN_MINLEN" \
#"$TRAIN_MAXLEN"
#done

for LANGUAGE in si en; do
    # we don't filter valid or test
    cp $TMP/valid.subword-nmt.$LANGUAGE $TMP/valid.subword-nmt.clean.$LANGUAGE
    cp $TMP/test.subword-nmt.$LANGUAGE $TMP/test.subword-nmt.clean.$LANGUAGE
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

#################################################
#   MOSES TOKENIZATION + LMVR (Ataman, 2017)    #
#################################################

echo "LMVR from Ataman (2017) ..."
echo "Not implemented!"
TMP=$DATA/wiki_${SRC}_${TGT}_lmvr
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_lmvr
mkdir -p "$TMP" "$DATABIN"

#################################################
#   MOSES TOKENIZATION + MORSEL (Lignos, 2010)  #
#################################################

echo "MORSEL from Lignos (2010) ..."
echo "Not implemented!"
TMP=$DATA/wiki_${SRC}_${TGT}_morsel
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_morsel
mkdir -p "$TMP" "$DATABIN"
