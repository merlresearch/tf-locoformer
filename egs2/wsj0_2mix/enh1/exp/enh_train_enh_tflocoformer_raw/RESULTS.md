<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: Apache-2.0
-->

<!-- Generated by ./scripts/utils/show_enh_score.sh -->
# RESULTS
## Environments
- date: `Mon Jun  3 10:48:32 EDT 2024`
- python version: `3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.1.0`
- Git hash: `90eed8e53498e7af682bc6ff39d9067ae440d6a4`
  - Commit date: `Mon May 27 22:42:15 2024 -0700`


## enh_train_enh_tflocoformer_raw

config: conf/tuning/train_enh_tflocoformer.yaml

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_cv_min_8k|98.18|23.61|23.28|35.28|22.98|
|enhanced_tt_min_8k|98.91|24.25|23.93|36.18|23.64|
