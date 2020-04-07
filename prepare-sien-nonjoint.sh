# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

SRC=si
TGT=en

BPESIZE=$1
if [ -z $BPESIZE ]
then
    BPESIZE=5000
fi

echo "BPE size = "$BPESIZE

TRAIN_MINLEN=6  # remove sentences with <6 BPE tokens
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=$ROOT/data
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}_nonjoint
mkdir -p $TMP $DATABIN

SRC_TOKENIZER="bash $SCRIPTS/indic_norm_tok.sh $SRC"
TGT_TOKENIZER="cat"  # learn target-side BPE over untokenized (raw) text
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
for ((i=0;i<${#URLS[@]};++i)); do
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

echo "pre-processing train data..."
bash $SCRIPTS/download_indic.sh
for FILE in "${TRAIN_SETS[@]}" ; do
    $SRC_TOKENIZER $DATA/$FILE.$SRC
done > $TMP/train.$SRC
for FILE in "${TRAIN_SETS[@]}"; do
    $TGT_TOKENIZER $DATA/$FILE.$TGT
done > $TMP/train.$TGT

echo "pre-processing dev/test data..."
$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC > $TMP/valid.$SRC
$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT > $TMP/valid.$TGT
$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC > $TMP/test.$SRC
$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT > $TMP/test.$TGT

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

# encode source side train/valid/test 
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.$SRC.bpe.model \
    --output_format=piece \
    --inputs $TMP/train.$SRC \
    --outputs $TMP/train.bpe.$SRC \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
for SPLIT in "valid" "test"; do \
    python $SPM_ENCODE \
        --model $DATABIN/sentencepiece.$SRC.bpe.model \
        --output_format=piece \
        --inputs $TMP/$SPLIT.$SRC 
        --outputs $TMP/$SPLIT.bpe.$SRC 
done

# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $TMP/train.bpe.$SRC \
  --validpref $TMP/valid.bpe.$SRC \
  --testpref $TMP/test.bpe.$SRC \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4

# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $TMP/train.bpe.$TGT \
  --validpref $TMP/valid.bpe.$TGT \
  --testpref $TMP/test.bpe.$TGT \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4
