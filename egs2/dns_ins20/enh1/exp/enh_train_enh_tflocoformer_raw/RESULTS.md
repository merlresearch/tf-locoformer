<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: Apache-2.0
-->

<!-- Generated by ./scripts/utils/show_enh_score.sh -->
# RESULTS
## Environments
- date: `Tue Jun 25 15:59:55 EDT 2024`
- python version: `3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.1.0`
- Git hash: `90eed8e53498e7af682bc6ff39d9067ae440d6a4`
  - Commit date: `Mon May 27 22:42:15 2024 -0700`


## enh_train_enh_tflocoformer_raw

config: conf/tuning/train_enh_tflocoformer.yaml

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_tt_synthetic_no_reverb|98.79|23.35|23.35|0.00|23.23|
|enhanced_tt_synthetic_with_reverb|83.19|13.17|13.17|0.00|10.97|