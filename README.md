# FLoRes Low Resource MT Benchmark (fork)

This repository is my own fork of FAIR's FLoRes repository.
My main reason for forking was to create training and evaluation scripts that are well-automated.

#### Notes
- target-side bpe learned from untokenized text.
    - what about trying to tokenize?

## Reproduced results

### Odd error messages for non-joint bpe

**EDIT: 5/12/2020**

After running this inside the `$TMP` folder of `prepare-neen-nonjoint.sh`:

```
for kind in train valid test
do
  for lang in ne en
  do
    echo "size of ${kind}.bpe.${lang}: "$(wc -l $kind.bpe.$lang) 
    echo "size of ${kind}.${lang}: "$(wc -l $kind.$lang)
  done
done
```

I get:

```
size of train.bpe.ne: 563853 train.bpe.ne
size of train.ne: 563947 train.ne
size of train.bpe.en: 563860 train.bpe.en
size of train.en: 563947 train.en
size of valid.bpe.ne: 2559 valid.bpe.ne
size of valid.ne: 2559 valid.ne
size of valid.bpe.en: 2559 valid.bpe.en
size of valid.en: 2559 valid.en
size of test.bpe.ne: 2835 test.bpe.ne
size of test.ne: 2835 test.ne
size of test.bpe.en: 2835 test.bpe.en
size of test.en: 2835 test.en
```

which seems to imple that `valid` and `test` files have the correct size, but when segmenting `train` with BPE, lines are lost:
- `train.ne => train.bpe.ne`: 563947 - 563853 = 94 lines lost
- `train.en => train.bpe.en`: 563947 - 563860 = 87 lines lost

In the "Encode with BPE" output for source:

```
processed 10000 lines
processed 20000 lines
[...]
processed 550000 lines
processed 560000 lines
skipped 0 empty lines
filtered 94 lines
skipped 0 empty lines
filtered 0 lines
skipped 0 empty lines
filtered 0 lines
```

and for target:

```
processed 540000 lines
processed 550000 lines
processed 560000 lines
skipped 0 empty lines
filtered 87 lines
skipped 0 empty lines
filtered 0 lines
skipped 0 empty lines
filtered 0 lines
```

**Question:** where is this filtering behavior coming from?

---

In the case of running non-joint prep script, seems like somehow the source/target datasets end up being of different length,
and the source dict is smaller

```
./log/2020-04-08T13-15-04-00-exp26-bpe5000-seed10-nonjoint
./checkpoints/2020-04-08T13-15-04-00-exp26-bpe5000-seed10-nonjoint

CUDA device is: 1
Logging output to: ./log/2020-04-08T13-15-04-00-exp26-bpe5000-seed10-nonjoint/baseline_ne_en.log
CUDA device is: 1
Logging output to: ./log/2020-04-08T13-15-04-00-exp26-bpe5000-seed10-nonjoint/baseline_en_ne.log
Traceback (most recent call last):
  File "/home/jonne/miniconda3/envs/flores/bin/fairseq-train", line 8, in <module>
    sys.exit(cli_main())
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq_cli/train.py", line 333, in cli_main
    main(args)
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq_cli/train.py", line 70, in main
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/checkpoint_utils.py", line 140, in load_checkpoint
    epoch=0, load_dataset=True, **passthrough_args
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/trainer.py", line 283, in get_train_iterator
    epoch=epoch,
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/tasks/fairseq_task.py", line 145, in get_batch_iterator
    indices = dataset.ordered_indices()
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/data/language_pair_dataset.py", line 273, in ordered_indices
    indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
IndexError: index 563856 is out of bounds for axis 0 with size 563853

CUDA device is: 1
Logging output to: ./log/2020-04-08T13-15-04-00-exp26-bpe5000-seed10-nonjoint/baseline_si_en.log
CUDA device is: 1
Logging output to: ./log/2020-04-08T13-15-04-00-exp26-bpe5000-seed10-nonjoint/baseline_en_si.log
Traceback (most recent call last):
  File "/home/jonne/miniconda3/envs/flores/bin/fairseq-train", line 8, in <module>
    sys.exit(cli_main())
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq_cli/train.py", line 333, in cli_main
    main(args)
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq_cli/train.py", line 70, in main
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/checkpoint_utils.py", line 140, in load_checkpoint
    epoch=0, load_dataset=True, **passthrough_args
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/trainer.py", line 283, in get_train_iterator
    epoch=epoch,
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/tasks/fairseq_task.py", line 145, in get_batch_iterator
    indices = dataset.ordered_indices()
  File "/home/jonne/miniconda3/envs/flores/lib/python3.7/site-packages/fairseq/data/language_pair_dataset.py", line 273, in ordered_indices
    indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
IndexError: index 427496 is out of bounds for axis 0 with size 421307
```

### Random seeds & different BPE settings

#### BPE=2500

```
### Raw results
      en-ne  en-si  ne-en  si-en
seed                            
10     4.42   1.64   7.59   6.69
11     4.33   1.04   7.56   6.43
12     4.51   1.28   7.51   6.42
13     4.44   0.93   7.46   6.35
14     4.40   1.65   7.50   6.54
15     4.60   1.86   7.15   6.33
16     4.34   1.49   7.63   6.37
17     4.64   1.53   7.75   6.39
18     4.52   1.41   7.52   7.10
19     4.39   1.72   7.60   6.48

### Summary statistics
        mean    std    25%    50%    75%
en-ne  4.459  0.105  4.392  4.430  4.518
en-si  1.455  0.297  1.312  1.510  1.647
ne-en  7.527  0.156  7.502  7.540  7.597
si-en  6.510  0.233  6.375  6.425  6.525

### Confidence interval
        mean    std     lb     ub
en-ne  4.459  0.105  4.249  4.669
en-si  1.455  0.297  0.861  2.049
ne-en  7.527  0.156  7.215  7.839
si-en  6.510  0.233  6.044  6.976
```

#### BPE=5000

```
### Raw results
      en-ne  en-si  ne-en  si-en
seed                            
10     4.29   1.00   7.83   6.90
11     4.54   1.41   7.33   6.42
12     4.61   1.12   7.91   6.56
13     4.52   0.95   7.76   6.70
14     4.53   1.37   7.89   6.61
15     4.57   1.42   7.81   6.47
16     4.55   1.49   7.61   6.45
17     4.42   1.07   7.82   6.36
18     4.58   1.51   7.84   6.95
19     4.52   0.81   7.80   6.44

### Summary statistics
        mean    std    25%    50%    75%
en-ne  4.513  0.093  4.520  4.535  4.565
en-si  1.215  0.253  1.018  1.245  1.418
ne-en  7.760  0.172  7.770  7.815  7.838
si-en  6.586  0.205  6.442  6.515  6.678

### Confidence interval
        mean    std     lb     ub
en-ne  4.513  0.093  4.327  4.699
en-si  1.215  0.253  0.709  1.721
ne-en  7.760  0.172  7.416  8.104
si-en  6.586  0.205  6.176  6.996
```

### Exploring different settings for BPE

- Note: all settings with `seed=10`

| Lang. pair | Reported |  BPE=2500   |  BPE=5000   |  BPE=7500  |
|------------|----------|-------------|-------------|------------|
|   EN-NE    |   4.3    |    4.39     |     4.52    |    4.44    |
|   NE-EN    |   7.6    |    7.60     |     7.80    |    7.44    |
|   EN-SI    |   1.2    |    1.72     |     0.81    |    1.32    |
|   SI-EN    |   7.2    |    6.48     |     6.44    |    6.98    |

#### BPE=2500

```
==== RECOVERING RESULTS FOR ./evaluate/2020-03-27T11-38-04-00-exp22-bpe2500/ ====
===== EVAL & LOG FILES =====
Eval file: ./evaluate/2020-03-27T11-38-04-00-exp22-bpe2500//baseline_ne_en.log
Log file: ./log/2020-03-26T12-39-04-00-exp22-bpe2500/baseline_ne_en.log
===== HYPERPARAMETERS =====
clip_norm=0.1
fixed_validation_seed=None
fp16=True
lr=[0.001]
max_tokens=4000
memory_efficient_fp16=False
min_lr=1e-09
seed=19
===== RESULTS =====
en-ne | 4.39
en-si | 1.72
ne-en | 7.60
si-en | 6.48
```

#### BPE=5000

```
==== RECOVERING RESULTS FOR ./evaluate/2020-03-25T13-08-04-00 ====
===== EVAL & LOG FILES =====
Eval file: ./evaluate/2020-03-25T13-08-04-00/baseline_ne_en.log
Log file: ./log/2020-03-23T06-33-04-00/baseline_ne_en.log
===== HYPERPARAMETERS =====
clip_norm=0.1
fixed_validation_seed=None
fp16=True
lr=[0.001]
max_tokens=4000
memory_efficient_fp16=False
min_lr=1e-09
seed=19
===== RESULTS =====
en-ne | 4.52
en-si | 0.81
ne-en | 7.80
si-en | 6.44
```

#### BPE=7500

```
==== RECOVERING RESULTS FOR ./evaluate/2020-03-28T00-47-04-00-exp23-bpe7500 ====
===== EVAL & LOG FILES =====
Eval file: ./evaluate/2020-03-28T00-47-04-00-exp23-bpe7500/baseline_ne_en.log
Log file: ./log/2020-03-27T12-58-04-00-exp23-bpe7500/baseline_ne_en.log
===== HYPERPARAMETERS =====
clip_norm=0.1
fixed_validation_seed=None
fp16=True
lr=[0.001]
max_tokens=4000
memory_efficient_fp16=False
min_lr=1e-09
seed=19
===== RESULTS =====
en-ne | 4.44
en-si | 1.32
ne-en | 7.44
si-en | 6.98
```

### Random seed experiments

- Notes
    - tried to run these in `Makefile` but some experiments weirdly did not get evaluated using `make evaluate_all` after training
        - cause: unclear
    - used `git grep` to find the log files with seeds that didn't get evalauted, and ran `evaluate_manual.sh` to produce BLEU scores for them
    - using `recover_results.sh`, created `evaluate/seed_results` which contains all the results for all the seed settings
    - wrote a parsing / analysis script `analyze_seed_results.py` that produces the tables below
        - interestingly it seems like there is more variability between seeds for the `en-si` / `si-en` pairs than `ne-en` / `en-ne`.

```
### Raw results
      en-ne  en-si  ne-en  si-en
seed                            
10     4.29   1.00   7.83   6.90
11     4.54   1.41   7.33   6.42
12     4.61   1.12   7.91   6.56
13     4.52   0.95   7.76   6.70
14     4.53   1.37   7.89   6.61
15     4.57   1.42   7.81   6.47
16     4.55   1.49   7.61   6.45
17     4.42   1.07   7.82   6.36
18     4.58   1.51   7.84   6.95
19     4.52   0.81   7.80   6.44

### Summary statistics
        mean    std    25%    50%    75%
en-ne  4.513  0.093  4.520  4.535  4.565
en-si  1.215  0.253  1.018  1.245  1.418
ne-en  7.760  0.172  7.770  7.815  7.838
si-en  6.586  0.205  6.442  6.515  6.678

### Confidence interval
        mean    std     lb     ub
en-ne  4.513  0.093  4.327  4.699
en-si  1.215  0.253  0.709  1.721
ne-en  7.760  0.172  7.416  8.104
si-en  6.586  0.205  6.176  6.996
```

### Recap before random seed experiments

```
Lang. pair             EN-NE NE-EN EN-SI SI-EN
Reported                 4.3   7.6   1.2   7.2
AWS/Azure               4.69  7.66  1.48  6.94
Brandeis                4.58  7.74  1.31  6.77
FP16                    4.59  7.39  1.24  6.69
FP16 + LB               3.99   7.1  1.06  5.82
FP16 + LB + LR=5e-4     4.03  6.48  1.13  5.39
FP16 + LB + LR=7e-4     4.04  6.87   1.1  5.65
FP16 + LB + seed12345   4.33  6.97  1.41  6.13
FP16 + BS20k            4.17  7.02  1.53     6
FP16+LB+CN0.1           4.03  7.34  1.05  6.08
FP16+LB+CN=0.1+LR=3e-3  3.03  6.11  0.75  6.03
FP16+LB+min_lr0.1       3.99   7.1  1.06  5.82
```

- Comparison to FP16+LB
    - Let's take the overall results table and diff each row with the FP16+LB experimental condition (use pandas for this)

```
Lang. pair             EN-NE NE-EN EN-SI SI-EN
FP16 + LB                  0     0     0     0
FP16 + LB + LR=5e-4     0.04 -0.62  0.07 -0.43
FP16 + LB + LR=7e-4     0.05 -0.23  0.04 -0.17
FP16 + LB + seed12345   0.34 -0.13  0.35  0.31
FP16 + BS20k            0.18 -0.08  0.47  0.18
FP16+LB+CN0.1           0.04  0.24 -0.01  0.26
FP16+LB+CN=0.1+LR=3e-3 -0.96 -0.99 -0.31  0.21
FP16+LB+min_lr0.1          0     0     0     0
```

- Overall notes (after exp11)
    - FP16 by itself does not hurt
    - going for larger batch seems to hurt all languages (quite a bit)
    - lowering the learning rate to `5e-4` or `7e-4` from the original `1e-3` only marginally improves EN -> X translation but hurts X -> EN much more
    - going even larger with the batch size (20k) seems to improve EN -> X translation and hurt X -> EN translation a bit.
    - `clip_norm=0.1` seems to improve everythin except EN-SI but even there the decrease is minimal. so maybe it's a good thing to use?
    - increasing the MINIMUM learning rate seems to do nothing
    - overall, messing with the actual learning rate seems dangerous


### FP16 + LB + default LR + min_lr=1e-8

| Lang. pair | Reported | Reproduced  | Difference  | Cloud provider |
|------------|----------|-------------|-------------|----------------|
|   EN-NE    |   4.3    |    3.99     |    -1.27    |   Brandeis     |
|   NE-EN    |   7.6    |    7.10     |    -1.49    |   Brandeis     |
|   EN-SI    |   1.2    |    1.06     |    -0.45    |   Brandeis     |
|   SI-EN    |   7.2    |    5.82     |    -1.17    |   Brandeis     |

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
=======
<p align="center">
<img src="flores_logo.png" width="500">
</p>

--------------------------------------------------------------------------------

# Facebook Low Resource MT Benchmark (FLoRes)
FLoRes is a benchmark dataset for machine translation between English and four low resource languages, Nepali, Sinhala, Khmer and Pashto, based on sentences translated from Wikipedia.
The data sets can be downloaded [HERE](https://github.com/facebookresearch/flores/raw/master/data/flores_test_sets.tgz).

**New**: two new languages, Khmer and Pashto, are added to the dataset.

This repository contains data and baselines from the paper:  
[The FLoRes Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English](https://arxiv.org/abs/1902.01382).

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
- 2020-04-02: Add two new langauge pairs, Khmer-English, Pashto-English.
- 2019-11-04: Add config to reproduce iterative back-translation result on Sinhala-English and English-Sinhala.
- 2019-10-23: Add script to reproduce iterative back-translation result on Nepali-English and English-Nepali.
- 2019-10-18: Add final test set.
- 2019-05-20: Remove extra carriage return character from Nepali-English parallel dataset.
- 2019-04-18: Specify the linebreak character in the sentencepiece encoding script to fix small portion of misaligned parallel sentences in Nepali-English parallel dataset.
- 2019-03-08: Update tokenizer script to make it compatible with previous version of indic_nlp.
- 2019-02-14: Update dataset preparation script to avoid unexpected extra line being added to each paralel dataset.


## License
The dataset is licenced under CC-BY-SA, see the LICENSE file for details.
