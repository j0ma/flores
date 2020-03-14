# FLoRes Low Resource MT Benchmark (fork)

This repository is my own fork of FAIR's FLoRes repository.

My main reason for forking was to create training and evaluation scripts that are well-automated.

## Reproduced results

### All settings so far

| Lang. pair | Reported |  AWS/Azure  | Brandeis |  FP16  |  FP16 + LB | FP16 + LB + LR=5e-4 | FP16 + LB + LR=7e-4 | FP16 + LB + seed12345 | FP16 + BS20k | FP16+LB+CN0.1 | FP16+LB+CN=0.1+LR=3e-3 |
|------------|----------|-------------|----------|--------|------------|---------------------|---------------------|-----------------------|--------------|---------------|------------------------|
|   EN-NE    |   4.3    |    4.69     |  4.58    |  4.59  |    3.99    |       4.03          |       4.04          |        4.33           |     4.17     |     4.03      |   3.03                 |
|   NE-EN    |   7.6    |    7.66     |  7.74    |  7.39  |    7.10    |       6.48          |       6.87          |        6.97           |     7.02     |     7.34      |   6.11                 |
|   EN-SI    |   1.2    |    1.48     |  1.31    |  1.24  |    1.06    |       1.13          |       1.10          |        1.41           |     1.53     |     1.05      |   0.75                 |
|   SI-EN    |   7.2    |    6.94     |  6.77    |  6.69  |    5.82    |       5.39          |       5.65          |        6.13           |     6.00     |     6.08      |   6.03                 |

### FP16 + LB + `clip_norm=0.1` + `lr=3e-3`

- log files erroneously say `lr=5e-3`

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    3.03     |    -1.27    |   Brandeis     |
|   NE-EN    |   7.6    |    6.11     |    -1.49    |   Brandeis     |
|   EN-SI    |   1.2    |    0.75     |    -0.45    |   Brandeis     |
|   SI-EN    |   7.2    |    6.03     |    -1.17    |   Brandeis     |

### FP16 + LB + `clip_norm=0.1` + `lr=5e-3`

- SI-EN and EN-SI work just fine
- NE-EN and EN-NE seem to have a problem with the loss scale

```
FloatingPointError: Minimum loss scale reached (0.0001). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.
```

### FP16 + Large batch + clip_norm=0.1

- `batch size=16000`
- `clip_norm=0.1`

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.03     |    -0.27    |   Brandeis     |
|   NE-EN    |   7.6    |    7.34     |    -0.26    |   Brandeis     |
|   EN-SI    |   1.2    |    1.05     |    -0.15    |   Brandeis     |
|   SI-EN    |   7.2    |    6.08     |    -1.12    |   Brandeis     |

### Batch size 20k
- Regular learning rate: `1e-3`
- FP16 training

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.17     |    -0.13    |   Brandeis     |
|   NE-EN    |   7.6    |    7.02     |    -0.58    |   Brandeis     |
|   EN-SI    |   1.2    |    1.53     |    0.31     |   Brandeis     |
|   SI-EN    |   7.2    |    6.00     |    -1.20    |   Brandeis     |

Training time: 8.79 hours on single Titan RTX

### Larger batch size, random seed
- Like first large batch experiment below, except with a random seed of `12345`
- Learning rate default `1e-3`

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.33     |    0.03     |   Brandeis     |
|   NE-EN    |   7.6    |    6.97     |    -0.63    |   Brandeis     |
|   EN-SI    |   1.2    |    1.41     |    0.21     |   Brandeis     |
|   SI-EN    |   7.2    |    6.13     |    -1.07    |   Brandeis     |

Training time: 8.91 hours on single Titan RTX

### Larger batch size, FP16, learning rate 7e-4
- Batch size enlarged using `--max-tokens 16000` 
- FP16 `--fp16`
- Removed `--update_freq 4`
- Learning rate set to `7e-14`

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.04     |    -0.26    |   Brandeis     |
|   NE-EN    |   7.6    |    6.87     |    -0.73    |   Brandeis     |
|   EN-SI    |   1.2    |    1.10     |    -0.10    |   Brandeis     |
|   SI-EN    |   7.2    |    5.65     |    -1.55    |   Brandeis     |

Training time: 8.91 hours on single Titan RTX

### Larger batch size, with FP16, learning rate 5e-4
- Batch size enlarged using `--max-tokens 16000` 
- FP16 `--fp16`
- Removed `--update_freq 4`
- Learning rate first set to `--lr 5e-3` -> error
- Then set learning rate to `--lr 5e-4` -> no error

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.03     |    -0.27    |   Brandeis     |
|   NE-EN    |   7.6    |    6.48     |    -1.12    |   Brandeis     |
|   EN-SI    |   1.2    |    1.13     |    -0.07    |   Brandeis     |
|   SI-EN    |   7.2    |    5.39     |    -1.81    |   Brandeis     |

Training time: 8.91 hours on single Titan RTX

### Larger batch size, with FP16
- Batch size enlarged using `--max-tokens 16000` 
- FP16 `--fp16`
- Removed `--update_freq 4`

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    3.99     |    -0.31    |   Brandeis     |
|   NE-EN    |   7.6    |    7.10     |    -0.21    |   Brandeis     |
|   EN-SI    |   1.2    |    1.06     |    -0.14    |   Brandeis     |
|   SI-EN    |   7.2    |    5.82     |    -1.38    |   Brandeis     |

Training time: 8.91 hours on single Titan RTX

### Reproduction on Brandeis hardware & FP16 training

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.59     |    0.29     |   Brandeis     |
|   NE-EN    |   7.6    |    7.39     |    -0.21    |   Brandeis     |
|   EN-SI    |   1.2    |    1.24     |    0.04     |   Brandeis     |
|   SI-EN    |   7.2    |    6.69     |    -0.51    |   Brandeis     |

Training time: 12.59 hours on single Titan RTX

### Reproduction on Brandeis hardware vol 2

Second re-run on Brandeis hardware to investigate whether there is randomness between runs on same GPU

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.58     |    0.28     |   Brandeis     |
|   NE-EN    |   7.6    |    7.74     |    0.14     |   Brandeis     |
|   EN-SI    |   1.2    |    1.31     |    0.11     |   Brandeis     |
|   SI-EN    |   7.2    |    6.77     |    -0.43    |   Brandeis     |

Training time: ~26 hours on single Titan RTX

### Reproduction on Brandeis hardware

In the next reproduction, I used a Titan RTX GPU, which decreased the training time to about 5 minutes per epoch.

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    4.58     |    0.28     |   Brandeis     |
|   NE-EN    |   7.6    |    7.74     |    0.14     |   Brandeis     |
|   EN-SI    |   1.2    |    1.31     |    0.11     |   Brandeis     |
|   SI-EN    |   7.2    |    6.77     |    -0.43    |   Brandeis     |

Training time: ~26 hours on single Titan RTX

Interestingly, evaluation is just as slow here as it was on Azure/AWS.

### Reproduction on AWS/Azure

In all experiments, the GPU used was a Tesla K80. Overall, I ran everything for 100 epochs, with about 20 min per epoch being the average runtime. 

| Lang. pair | Reported | Reproduced | Difference | Cloud provider |
|------------|----------|------------|------------|----------------|
|   EN-NE    |   4.3    |    4.69    |    0.39    |     AWS        |
|   NE-EN    |   7.6    |    7.66    |    0.06    |    Azure       |
|   EN-SI    |   1.2    |    1.48    |    0.28    |     AWS        |
|   SI-EN    |   7.2    |    6.94    |    -0.26   |     AWS        |

## Notes
- old log files from azure/aws located in `./log/old_from_awsazure`
- old eval files located in `evaluate/old-eval-scripts` 
- old eval results located in `evalute/aws-azure-results`

---

# FLoRes Low Resource MT Benchmark

This repository contains data and baselines from the paper:  
[The FLoRes Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English](https://arxiv.org/abs/1902.01382).

The data can be downloaded directly at:  
https://github.com/facebookresearch/flores/raw/master/data/wikipedia_en_ne_si_test_sets.tgz

## Baselines

The following instructions will can be used to reproduce the baseline results from the paper.

### Requirements

The baseline uses the
[Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) and
[sentencepiece](https://github.com/google/sentencepiece) for preprocessing;
[fairseq](https://github.com/pytorch/fairseq) for model training; and
[sacrebleu](https://github.com/mjpost/sacreBLEU) for scoring.

Dependencies can be installed via pip:
```
$ pip install fairseq sacrebleu sentencepiece
```

The Indic NLP Library will be cloned automatically by the `prepare-{ne,si}en.sh` scripts.

### Download and preprocess data

The `download-data.sh` script can be used to download and extract the raw data.
Thereafter the `prepare-neen.sh` and `prepare-sien.sh` scripts can be used to
preprocess the raw data. In particular, they will use the sentencepiece library
to learn a shared BPE vocabulary with 5000 subword units and binarize the data
for training with fairseq.

To download and extract the raw data:
```
$ bash download-data.sh
```

Thereafter, run the following to preprocess the raw data:
```
$ bash prepare-neen.sh
$ bash prepare-sien.sh
```

### Train a baseline Transformer model

To train a baseline Ne-En model on a single GPU:
```
$ CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/wiki_ne_en_bpe5000/ \
    --source-lang ne --target-lang en \
    --arch transformer --share-all-embeddings \
    --encoder-layers 5 --decoder-layers 5 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 2 --decoder-attention-heads 2 \
    --encoder-normalize-before --decoder-normalize-before \
    --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
    --weight-decay 0.0001 \
    --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --lr 1e-3 --min-lr 1e-9 \
    --max-tokens 4000 \
    --update-freq 4 \
    --max-epoch 100 --save-interval 10
```

To train on 4 GPUs, remove the `--update-freq` flag and run `CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train (...)`.
If you have a Volta or newer GPU you can further improve training speed by adding the `--fp16` flag.

This same architecture can be used for En-Ne, Si-En and En-Si:
- For En-Ne, update the training command with:  
  `fairseq-train data-bin/wiki_ne_en_bpe5000 --source-lang en --target-lang ne`
- For Si-En, update the training command with:  
  `fairseq-train data-bin/wiki_si_en_bpe5000 --source-lang si --target-lang en`
- For En-Si, update the training command with:  
  `fairseq-train data-bin/wiki_si_en_bpe5000 --source-lang en --target-lang si`

### Compute BLEU using sacrebleu

Run beam search generation and scoring with sacrebleu:
```
$ fairseq-generate \
    data-bin/wiki_ne_en_bpe5000/ \
    --source-lang ne --target-lang en \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 --lenpen 1.2 \
    --gen-subset valid \
    --remove-bpe=sentencepiece \
    --sacrebleu
```

Note that the `--gen-subset valid` set is the FloRes **dev** set and `--gen-subset test` set is the FloRes **devtest** set.
Replace `--gen-subset valid` with `--gen-subset test` above to score the FLoRes **devtest** set which is corresponding to the reported number in our paper.

**Tokenized BLEU for En-Ne and En-Si:**

For these language pairs we report tokenized BLEU. You can compute tokenized BLEU by removing the `--sacrebleu` flag
from generate.py:
```
$ fairseq-generate \
    data-bin/wiki_ne_en_bpe5000/ \
    --source-lang en --target-lang ne \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 --lenpen 1.2 \
    --gen-subset valid \
    --remove-bpe=sentencepiece
```

### Train iterative back-translation models

After runing the commands in *Download and preprocess data* section above, run the following to download and preprocess the monolingual data:
```
$ bash prepare-monolingual.sh
```

To train the iterative back-translation for two iterations on Ne-En, run the following:
```
$ bash reproduce.sh ne_en
```

The script will train an Ne-En supervised model, translate Nepali monolingual data, train En-Ne back-translation iteration 1 model, translate English monolingual data back to Nepali, and train Ne-En back-translation iteration 2 model. All the model training and data generation happen locally. The script uses all the GPUs listed in `CUDA_VISIBLE_DEVICES` variable unless certain cuda device ids are specified to `train.py`, and it is designed to adjust the hyper-parameters according to the number of available GPUs.  With 8 Tesla V100 GPUs, the full pipeline takes about 25 hours to finish. We expect the final BT iteration 2 Ne-En model achieves around 15.9 (sacre)BLEU score on devtest set. The script supports `ne_en`, `en_ne`, `si_en` and `en_si` directions.

## Citation

If you use this data in your work, please cite:

```bibtex
@inproceedings{,
  title={Two New Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English},
  author={Guzm\'{a}n, Francisco and Chen, Peng-Jen and Ott, Myle and Pino, Juan and Lample, Guillaume and Koehn, Philipp and Chaudhary, Vishrav and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:1902.01382},
  year={2019}
}
```

## Changelog
- 2019-11-04: Add config to reproduce iterative back-translation result on Sinhala-English and English-Sinhala
- 2019-10-23: Add script to reproduce iterative back-translation result on Nepali-English and English-Nepali
- 2019-10-18: Add final test set
- 2019-05-20: Remove extra carriage return character from Nepali-English parallel dataset.
- 2019-04-18: Specify the linebreak character in the sentencepiece encoding script to fix small portion of misaligned parallel sentences in Nepali-English parallel dataset.
- 2019-03-08: Update tokenizer script to make it compatible with previous version of indic_nlp.
- 2019-02-14: Update dataset preparation script to avoid unexpected extra line being added to each paralel dataset.


## License
The dataset is licenced under CC-BY-SA, see the LICENSE file for details.
