# FLoRes Low Resource MT Benchmark (fork)

This repository is my own fork of FAIR's FLoRes repository.
My main reason for forking was to create training and evaluation scripts that are well-automated.

## Notes
- manually removed extraneous empty line from lmvr tuned segmentations in the interest of time (8/9/20)

- wrote script `remove_slashes.py` to remove slashes in lmvr tuned segmentations (2/11/21)

- edited `download_segmentation_models.sh` and wrote script `update_kk_segmentations.sh` to switch to 225k segmentations without having to edit a bunch of paths

### WMT19 KK-EN

```
### Raw results
                  bleu
pair             en-kk kk-en
method      seed
baseline    10     0.8   2.0
            11     0.7   2.2
            12     0.6   2.3
            13     0.6   2.0
            14     0.8   1.9
subword-nmt 10     0.8   2.7
            11     0.9   2.8
            12     0.9   2.9
            13     0.9   2.8
            14     0.9   2.9

### Summary statistics
                   count  mean    std  25%  50%  75%
method      pair
baseline    en-kk    5.0  0.70  0.100  0.6  0.7  0.8
            kk-en    5.0  2.08  0.164  2.0  2.0  2.2
subword-nmt en-kk    5.0  0.88  0.045  0.9  0.9  0.9
            kk-en    5.0  2.82  0.084  2.8  2.8  2.9

### Confidence interval
                   mean    std  std_err     lb     ub
method      pair
baseline    en-kk  0.70  0.100    0.045  0.611  0.789
            kk-en  2.08  0.164    0.073  1.933  2.227
subword-nmt en-kk  0.88  0.045    0.020  0.840  0.920
            kk-en  2.82  0.084    0.037  2.745  2.895

### Reported results from paper
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2
dtype: float64
```

### Experimental results as of 8/9/2020


```
### Raw results
                    bleu
pair               en-ne en-si ne-en si-en
method        seed
baseline      10     4.6   1.2   8.3   7.3
              11     4.4   1.4   8.5   7.5
              12     4.6   0.8   8.3   7.3
              13     4.5   1.0   8.3   7.5
              14     4.6   1.0   8.1   7.3
baseline-fp16 10     4.4   0.9   8.5   7.4
              11     4.8   1.6   8.7   7.6
              12     4.5   1.0   8.3   7.5
              13     4.6   0.8   8.5   7.5
              14     4.6   1.1   8.1   7.6
lmvr          10     4.0   1.1   8.1   7.0
              11     4.1   1.4   8.1   6.9
              12     4.1   1.0   8.0   7.2
              13     4.2   1.3   8.4   7.4
              14     4.3   1.2   7.9   7.2
lmvr-tuned    10     4.3   1.4   7.9   7.3
              11     4.3   1.8   7.8   6.9
              12     4.3   1.7   7.8   7.3
              13     4.3   1.0   7.7   7.2
              14     4.4   1.3   8.0   7.5
morsel        10     4.5   1.1   5.4   7.9
              11     4.3   1.2   5.6   8.0
              12     4.3   1.0   4.8   7.7
              13     4.6   1.0   5.3   7.6
              14     4.2   1.3   5.4   7.7
subword-nmt   10     4.2   0.8   8.6   7.7
              11     4.5   0.8   8.3   8.0
              12     4.6   0.8   8.5   7.7
              13     4.5   1.1   8.3   7.4
              14     4.3   0.9   8.6   8.0

### Summary statistics
                     count  mean    std  25%  50%  75%
method        pair
baseline      en-ne    5.0  4.54  0.089  4.5  4.6  4.6
              en-si    5.0  1.08  0.228  1.0  1.0  1.2
              ne-en    5.0  8.30  0.141  8.3  8.3  8.3
              si-en    5.0  7.38  0.110  7.3  7.3  7.5
baseline-fp16 en-ne    5.0  4.58  0.148  4.5  4.6  4.6
              en-si    5.0  1.08  0.311  0.9  1.0  1.1
              ne-en    5.0  8.42  0.228  8.3  8.5  8.5
              si-en    5.0  7.52  0.084  7.5  7.5  7.6
lmvr          en-ne    5.0  4.14  0.114  4.1  4.1  4.2
              en-si    5.0  1.20  0.158  1.1  1.2  1.3
              ne-en    5.0  8.10  0.187  8.0  8.1  8.1
              si-en    5.0  7.14  0.195  7.0  7.2  7.2
lmvr-tuned    en-ne    5.0  4.32  0.045  4.3  4.3  4.3
              en-si    5.0  1.44  0.321  1.3  1.4  1.7
              ne-en    5.0  7.84  0.114  7.8  7.8  7.9
              si-en    5.0  7.24  0.219  7.2  7.3  7.3
morsel        en-ne    5.0  4.38  0.164  4.3  4.3  4.5
              en-si    5.0  1.12  0.130  1.0  1.1  1.2
              ne-en    5.0  5.30  0.300  5.3  5.4  5.4
              si-en    5.0  7.78  0.164  7.7  7.7  7.9
subword-nmt   en-ne    5.0  4.42  0.164  4.3  4.5  4.5
              en-si    5.0  0.88  0.130  0.8  0.8  0.9
              ne-en    5.0  8.46  0.152  8.3  8.5  8.6
              si-en    5.0  7.76  0.251  7.7  7.7  8.0

### Confidence interval
                     mean    std  std_err     lb     ub
method        pair
baseline      en-ne  4.54  0.089    0.040  4.460  4.620
              en-si  1.08  0.228    0.102  0.876  1.284
              ne-en  8.30  0.141    0.063  8.174  8.426
              si-en  7.38  0.110    0.049  7.282  7.478
baseline-fp16 en-ne  4.58  0.148    0.066  4.447  4.713
              en-si  1.08  0.311    0.139  0.801  1.359
              ne-en  8.42  0.228    0.102  8.216  8.624
              si-en  7.52  0.084    0.037  7.445  7.595
lmvr          en-ne  4.14  0.114    0.051  4.038  4.242
              en-si  1.20  0.158    0.071  1.059  1.341
              ne-en  8.10  0.187    0.084  7.933  8.267
              si-en  7.14  0.195    0.087  6.966  7.314
lmvr-tuned    en-ne  4.32  0.045    0.020  4.280  4.360
              en-si  1.44  0.321    0.144  1.153  1.727
              ne-en  7.84  0.114    0.051  7.738  7.942
              si-en  7.24  0.219    0.098  7.044  7.436
morsel        en-ne  4.38  0.164    0.073  4.233  4.527
              en-si  1.12  0.130    0.058  1.003  1.237
              ne-en  5.30  0.300    0.134  5.032  5.568
              si-en  7.78  0.164    0.073  7.633  7.927
subword-nmt   en-ne  4.42  0.164    0.073  4.273  4.567
              en-si  0.88  0.130    0.058  0.763  0.997
              ne-en  8.46  0.152    0.068  8.324  8.596
              si-en  7.76  0.251    0.112  7.536  7.984

### Reported results from paper
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2

```


### Experimental results as of 7/13/2020

These are based on evaluation using `fairseq-interactive` and `sacrebleu`. All BLEU scores are case-insensitive.

```
### Raw results
                    bleu
pair               en-ne en-si ne-en si-en
method        seed
baseline      10     4.6   1.2   8.3   7.3
              11     4.4   1.4   8.5   7.5
              12     4.6   0.8   8.3   7.3
              13     4.5   1.0   8.3   7.5
              14     4.6   1.0   8.1   7.3
baseline-fp16 10     4.3   0.9   8.5   7.4
              11     4.7   1.6   8.7   7.6
              12     4.4   1.0   8.3   7.5
              13     4.6   0.8   8.5   7.5
              14     4.5   1.1   8.1   7.6
lmvr          10     3.9   1.1   8.1   7.0
              11     4.0   1.4   8.1   6.9
              12     4.0   1.0   8.0   7.2
              13     4.1   1.3   8.4   7.4
              14     4.2   1.2   7.9   7.2
subword-nmt   10     4.2   0.8   8.6   7.7
              11     4.5   0.8   8.3   8.0
              12     4.6   0.8   8.5   7.7
              13     4.5   1.1   8.3   7.4
              14     4.3   0.9   8.6   8.0

### Summary statistics
                     count  mean    std  25%  50%  75%
method        pair
baseline      en-ne    5.0  4.54  0.089  4.5  4.6  4.6
              en-si    5.0  1.08  0.228  1.0  1.0  1.2
              ne-en    5.0  8.30  0.141  8.3  8.3  8.3
              si-en    5.0  7.38  0.110  7.3  7.3  7.5
baseline-fp16 en-ne    5.0  4.50  0.158  4.4  4.5  4.6
              en-si    5.0  1.08  0.311  0.9  1.0  1.1
              ne-en    5.0  8.42  0.228  8.3  8.5  8.5
              si-en    5.0  7.52  0.084  7.5  7.5  7.6
lmvr          en-ne    5.0  4.04  0.114  4.0  4.0  4.1
              en-si    5.0  1.20  0.158  1.1  1.2  1.3
              ne-en    5.0  8.10  0.187  8.0  8.1  8.1
              si-en    5.0  7.14  0.195  7.0  7.2  7.2
subword-nmt   en-ne    5.0  4.42  0.164  4.3  4.5  4.5
              en-si    5.0  0.88  0.130  0.8  0.8  0.9
              ne-en    5.0  8.46  0.152  8.3  8.5  8.6
              si-en    5.0  7.76  0.251  7.7  7.7  8.0

### Confidence interval
                     mean    std  std_err     lb     ub
method        pair
baseline      en-ne  4.54  0.089    0.040  4.460  4.620
              en-si  1.08  0.228    0.102  0.876  1.284
              ne-en  8.30  0.141    0.063  8.174  8.426
              si-en  7.38  0.110    0.049  7.282  7.478
baseline-fp16 en-ne  4.50  0.158    0.071  4.359  4.641
              en-si  1.08  0.311    0.139  0.801  1.359
              ne-en  8.42  0.228    0.102  8.216  8.624
              si-en  7.52  0.084    0.037  7.445  7.595
lmvr          en-ne  4.04  0.114    0.051  3.938  4.142
              en-si  1.20  0.158    0.071  1.059  1.341
              ne-en  8.10  0.187    0.084  7.933  8.267
              si-en  7.14  0.195    0.087  6.966  7.314
subword-nmt   en-ne  4.42  0.164    0.073  4.273  4.567
              en-si  0.88  0.130    0.058  0.763  0.997
              ne-en  8.46  0.152    0.068  8.324  8.596
              si-en  7.76  0.251    0.112  7.536  7.984

### Reported results from paper
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2

```

---

#### Notes
- target-side bpe learned from untokenized text.
    - what about trying to tokenize?
- Makefile doesn't work / is not in use as of 5/19/20
- exp26 and exp27 refer pretty much to the same thing, i simply jumbled up the naming

## Reproduced results

### Nonjoint BPE experiments

#### BPE=2500

```
### Raw results
      en-ne  ne-en  en-si  si-en
seed
10     4.17   6.66   0.86   5.87
11     4.29   6.57   1.01   6.08
12     4.33   6.99   1.06   5.88
13     4.18   6.98   1.18   5.78
14     4.25   6.72   1.48   5.89
15     4.26   6.79   1.36   5.79
16     4.22   7.06   0.98   6.18
17     4.49   7.11   1.29   5.56
18     4.20   6.53   1.47   6.21
19     4.21   6.54   1.43   5.83

### Summary statistics
       count   mean    std    25%    50%    75%
en-ne   10.0  4.260  0.095  4.203  4.235  4.283
ne-en   10.0  6.795  0.224  6.592  6.755  6.988
en-si   10.0  1.212  0.225  1.022  1.235  1.412
si-en   10.0  5.907  0.199  5.800  5.875  6.033

### Confidence interval
        mean    std  std_err     lb     ub
en-ne  4.260  0.095    0.030  4.200  4.320
ne-en  6.795  0.224    0.071  6.653  6.937
en-si  1.212  0.225    0.071  1.070  1.354
si-en  5.907  0.199    0.063  5.781  6.033

### Reported results
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2
dtype: float64

### Difference from reported
      en-ne  ne-en  en-si  si-en
seed
10    -0.13  -0.94  -0.34  -1.33
11    -0.01  -1.03  -0.19  -1.12
12     0.03  -0.61  -0.14  -1.32
13    -0.12  -0.62  -0.02  -1.42
14    -0.05  -0.88   0.28  -1.31
15    -0.04  -0.81   0.16  -1.41
16    -0.08  -0.54  -0.22  -1.02
17     0.19  -0.49   0.09  -1.64
18    -0.10  -1.07   0.27  -0.99
19    -0.09  -1.06   0.23  -1.37

### Fraction of overestimates
en-ne    0.2
ne-en    0.0
en-si    0.5
si-en    0.0
dtype: float64
```

#### BPE=5000
- other settings as with joint bpe

```
### Raw results
      en-ne  en-si  ne-en  si-en
seed
10     4.06   1.15   7.17   6.49
11     4.14   0.70   7.08   6.10
12     4.30   0.81   6.43   6.28
13     4.29   1.29   7.03   6.22
14     4.42   0.95   6.73   6.14
15     4.06   1.00   7.16   6.14
16     4.37   0.93   7.26   6.28
17     4.26   1.12   6.97   6.14
18     4.61   1.01   7.30   6.35
19     4.34   1.20   7.07   6.13

### Summary statistics
       count   mean    std    25%    50%    75%
en-ne   10.0  4.285  0.169  4.170  4.295  4.362
en-si   10.0  1.016  0.180  0.935  1.005  1.142
ne-en   10.0  7.020  0.262  6.985  7.075  7.167
si-en   10.0  6.227  0.124  6.140  6.180  6.280

### Confidence interval
        mean    std  std_err     lb     ub
en-ne  4.285  0.169    0.053  4.178  4.392
en-si  1.016  0.180    0.057  0.902  1.130
ne-en  7.020  0.262    0.083  6.854  7.186
si-en  6.227  0.124    0.039  6.149  6.305

### Reported results
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2
dtype: float64

### Difference from reported
      en-ne  en-si  ne-en  si-en
seed
10    -0.24  -0.05  -0.43  -0.71
11    -0.16  -0.50  -0.52  -1.10
12     0.00  -0.39  -1.17  -0.92
13    -0.01   0.09  -0.57  -0.98
14     0.12  -0.25  -0.87  -1.06
15    -0.24  -0.20  -0.44  -1.06
16     0.07  -0.27  -0.34  -0.92
17    -0.04  -0.08  -0.63  -1.06
18     0.31  -0.19  -0.30  -0.85
19     0.04   0.00  -0.53  -1.07

### Fraction of overestimates
en-ne    0.4
en-si    0.1
ne-en    0.0
si-en    0.0
dtype: float64

```

#### BPE=7500

```
### Raw results
      en-si  si-en  en-ne  ne-en
seed
10     0.81   6.62   4.17   7.64
11     0.60   6.50   4.10   7.21
12     0.74   6.38   4.18   7.56
13     0.66   6.70   4.30   7.43
14     1.58   6.32   4.13   7.14
15     0.75   6.48   4.41   7.29
16     1.19   6.55   4.47   7.37
17     1.38   6.05   4.16   7.35
18     0.65   6.53   4.46   7.37
19     0.99   6.38   4.39   7.07

### Summary statistics
       count   mean    std    25%   50%    75%
en-si   10.0  0.935  0.340  0.680  0.78  1.140
si-en   10.0  6.451  0.182  6.380  6.49  6.545
en-ne   10.0  4.277  0.145  4.162  4.24  4.405
ne-en   10.0  7.343  0.177  7.230  7.36  7.415

### Confidence interval
        mean    std  std_err     lb     ub
en-si  0.935  0.340    0.108  0.720  1.150
si-en  6.451  0.182    0.058  6.336  6.566
en-ne  4.277  0.145    0.046  4.185  4.369
ne-en  7.343  0.177    0.056  7.231  7.455

### Reported results
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2
dtype: float64

### Difference from reported
      en-ne  en-si  ne-en  si-en
seed
10    -0.13  -0.39   0.04  -0.58
11    -0.20  -0.60  -0.39  -0.70
12    -0.12  -0.46  -0.04  -0.82
13     0.00  -0.54  -0.17  -0.50
14    -0.17   0.38  -0.46  -0.88
15     0.11  -0.45  -0.31  -0.72
16     0.17  -0.01  -0.23  -0.65
17    -0.14   0.18  -0.25  -1.15
18     0.16  -0.55  -0.23  -0.67
19     0.09  -0.21  -0.53  -0.82

### Fraction of overestimates
en-ne    0.4
en-si    0.2
ne-en    0.1
si-en    0.0
dtype: float64
```

Recall joint bpe:
```
      en-ne  en-si  ne-en  si-en
seed
10     4.42   1.64   7.59   6.69
```

### Random seeds with joint BPE

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
       count   mean    std    25%    50%    75%
en-ne   10.0  4.459  0.105  4.392  4.430  4.518
en-si   10.0  1.455  0.297  1.312  1.510  1.647
ne-en   10.0  7.527  0.156  7.502  7.540  7.597
si-en   10.0  6.510  0.233  6.375  6.425  6.525

### Confidence interval
        mean    std  std_err     lb     ub
en-ne  4.459  0.105    0.033  4.393  4.525
en-si  1.455  0.297    0.094  1.267  1.643
ne-en  7.527  0.156    0.049  7.428  7.626
si-en  6.510  0.233    0.074  6.363  6.657

### Reported results
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2
dtype: float64

### Difference from reported
      en-ne  en-si  ne-en  si-en
seed
10     0.12   0.44  -0.01  -0.51
11     0.03  -0.16  -0.04  -0.77
12     0.21   0.08  -0.09  -0.78
13     0.14  -0.27  -0.14  -0.85
14     0.10   0.45  -0.10  -0.66
15     0.30   0.66  -0.45  -0.87
16     0.04   0.29   0.03  -0.83
17     0.34   0.33   0.15  -0.81
18     0.22   0.21  -0.08  -0.10
19     0.09   0.52   0.00  -0.72

### Fraction of overestimates
en-ne    1.0
en-si    0.8
ne-en    0.2
si-en    0.0
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
       count   mean    std    25%    50%    75%
en-ne   10.0  4.513  0.093  4.520  4.535  4.565
en-si   10.0  1.215  0.253  1.018  1.245  1.418
ne-en   10.0  7.760  0.172  7.770  7.815  7.838
si-en   10.0  6.586  0.205  6.442  6.515  6.678

### Confidence interval
        mean    std  std_err     lb     ub
en-ne  4.513  0.093    0.029  4.454  4.572
en-si  1.215  0.253    0.080  1.055  1.375
ne-en  7.760  0.172    0.054  7.651  7.869
si-en  6.586  0.205    0.065  6.456  6.716

### Reported results
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2
dtype: float64

### Difference from reported
      en-ne  en-si  ne-en  si-en
seed
10    -0.01  -0.20   0.23  -0.30
11     0.24   0.21  -0.27  -0.78
12     0.31  -0.08   0.31  -0.64
13     0.22  -0.25   0.16  -0.50
14     0.23   0.17   0.29  -0.59
15     0.27   0.22   0.21  -0.73
16     0.25   0.29   0.01  -0.75
17     0.12  -0.13   0.22  -0.84
18     0.28   0.31   0.24  -0.25
19     0.22  -0.39   0.20  -0.76

### Fraction of overestimates
en-ne    0.9
en-si    0.5
ne-en    0.9
si-en    0.0
dtype: float64
```

#### BPE = 7500

```
### Raw results
      en-ne  en-si  ne-en  si-en
seed
10     4.57   1.07   7.52   7.40
11     4.51   1.00   7.38   6.69
12     4.55   0.88   7.26   6.66
13     4.58   0.84   7.74   6.88
14     4.54   1.26   7.31   6.79
15     4.51   0.74   7.26   6.96
16     4.58   1.18   7.76   6.76
17     4.24   1.32   7.43   6.85
18     4.45   1.12   7.47   6.65
19     4.44   1.32   7.44   6.98

### Summary statistics
       count   mean    std    25%    50%    75%
en-ne   10.0  4.497  0.103  4.465  4.525  4.565
en-si   10.0  1.073  0.205  0.910  1.095  1.240
ne-en   10.0  7.457  0.177  7.328  7.435  7.507
si-en   10.0  6.862  0.222  6.708  6.820  6.940

### Confidence interval
        mean    std  std_err     lb     ub
en-ne  4.497  0.103    0.033  4.432  4.562
en-si  1.073  0.205    0.065  0.943  1.203
ne-en  7.457  0.177    0.056  7.345  7.569
si-en  6.862  0.222    0.070  6.722  7.002

### Reported results
en-ne    4.3
ne-en    7.6
en-si    1.2
si-en    7.2
dtype: float64

### Difference from reported
      en-ne  en-si  ne-en  si-en
seed
10     0.27  -0.13  -0.08   0.20
11     0.21  -0.20  -0.22  -0.51
12     0.25  -0.32  -0.34  -0.54
13     0.28  -0.36   0.14  -0.32
14     0.24   0.06  -0.29  -0.41
15     0.21  -0.46  -0.34  -0.24
16     0.28  -0.02   0.16  -0.44
17    -0.06   0.12  -0.17  -0.35
18     0.15  -0.08  -0.13  -0.55
19     0.14   0.12  -0.16  -0.22

### Fraction of overestimates
en-ne    0.9
en-si    0.3
ne-en    0.2
si-en    0.1
dtype: float64
```

### Exploring different settings for BPE

- Note: all settings with `seed=19`

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
