# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

SRC=ne
TGT=en

BPESIZE=$1
if [ -z $BPESIZE ]
then
    BPESIZE=5000
fi

echo "BPE size = "$BPESIZE

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=$ROOT/data
TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}_lowercase
DATABIN=$ROOT/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}
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

# lowercase english side
$SCRIPTS/lowercase-english-side.sh $TMP

# rename nepali side
working_dir=$(pwd)
cd $TMP
for old_file in ./*.ne
do
    new_file=$(echo "$old_file" | sed "s/\.ne/\.lower.ne/g")
    echo "${old_file} -> ${new_file}"
    mv $old_file $new_file
done
cd $working_dir

# learn BPE with sentencepiece
python $SPM_TRAIN \
  --input=$TMP/train.lower.$SRC,$TMP/train.lower.$TGT \
  --model_prefix=$DATABIN/sentencepiece.bpe \
  --vocab_size=$BPESIZE \
  --character_coverage=1.0 \
  --model_type=bpe

# encode train/valid/test
python $SPM_ENCODE \
  --model $DATABIN/sentencepiece.bpe.model \
  --output_format=piece \
  --inputs $TMP/train.lower.$SRC $TMP/train.lower.$TGT \
  --outputs $TMP/train.lower.bpe.$SRC $TMP/train.lower.bpe.$TGT \
  --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
for SPLIT in "valid" "test"; do \
  python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.bpe.model \
    --output_format=piece \
    --inputs $TMP/$SPLIT.lower.$SRC $TMP/$SPLIT.lower.$TGT \
    --outputs $TMP/$SPLIT.lower.bpe.$SRC $TMP/$SPLIT.lower.bpe.$TGT
done

# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $TMP/train.lower.bpe \
  --validpref $TMP/valid.lower.bpe \
  --testpref $TMP/test.lower.bpe \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4
