# Negative-sampling ratio ablation

negative_sampling.ratio 를 변경해 dataset 을 다시 빌드한 뒤 동일
feature 셋과 LightGBM hyperparam 으로 5-fold OOF 성능을 비교했습니다.

|   ratio |   n_pos |   n_neg |   pos_rate |   oof_auc |   oof_pr_auc |
|--------:|--------:|--------:|-----------:|----------:|-------------:|
|       1 |    2295 |    2295 |      0.5   |    0.9324 |       0.9101 |
|       3 |    2295 |    6885 |      0.25  |    0.9427 |       0.8145 |
|       5 |    2295 |   11475 |      0.167 |    0.946  |       0.7424 |
|      10 |    2295 |   22950 |      0.091 |    0.9511 |       0.6296 |
