<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: Apache-2.0
-->

# TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation and Enhancement

This repository includes source code of the TF-Locoformer model proposed in the following paper:

```
@InProceedings{Saijo2024_TFLoco,
  author    =  {Saijo, Kohei and Wichern, Gordon and Germain, Fran\c{c}ois G. and Pan, Zexu and {Le Roux}, Jonathan},
  title     =  {TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation and Enhancement},
  booktitle =  {Proc. International Workshop on Acoustic Signal Enhancement (IWAENC)},
  year      =  2024,
  month     =  sep
}
```

## Table of contents

1. [Environmental setup: Installing ESPnet from source and injecting the TF-Locoformer code](#environmental-setup-installing-espnet-from-source-and-injecting-the-tf-locoformer-code)
2. [Using a pre-trained model](#using-a-pre-trained-model)
3. [Example of training and inference](#example-of-training-and-inference)
4. [Instructions for running training on each dataset in the ESPnet pipeline (Librimix, WHAMR! and DNS)](#instructions-for-running-training-on-each-dataset-in-the-espnet-pipeline)
5. [Contributing](#contributing)
6. [Copyright and license](#copyright-and-license)

## Environmental setup: Installing ESPnet from source and injecting the TF-Locoformer code

In this repo, we provide the code for TF-Locoformer along with scripts to run training and inference in ESPnet.
The following commands install ESPnet from source and copy the TF-Locoformer code to the appropriate directories in ESPnet.

For more details on installing ESPnet, please refer to https://espnet.github.io/espnet/installation.html.

```sh
# Clone espnet code.
git clone https://github.com/espnet/espnet.git

# Checkout the commit where we tested our code.
cd ./espnet && git checkout 90eed8e53498e7af682bc6ff39d9067ae440d6a4

# Set up conda environment.
# ./setup_anaconda /path/to/conda environment-name python-version
cd ./tools && ./setup_anaconda.sh /path/to/conda tflocoformer 3.10.8

# Install espnet from source with other dependencies. We used torch 2.1.0 and cuda 11.8.
# NOTE: torch version must be 2.x.x for other dependencies.
make TH_VERSION=2.1.0 CUDA_VERSION=11.8

# Install the RoPE package.
conda activate tflocoformer && pip install rotary-embedding-torch==0.6.1

# Copy the TF-Locoformer code to ESPnet.
# NOTE: ./copy_files_to_espnet.sh changes `espnet2/tasks/enh.py`. Please be careful when using your existing ESPnet environment.
cd ../../ && git clone https://github.com/merl-oss-private/tf-locoformer.git && cd tf_locoformer
./copy_files_to_espnet.sh /path/to/espnet-root
```

## Using a pre-trained model

This repo supports speech separation/enhancement on 4 datasets:

- WSJ0-2mix (`egs2/wsj0_2mix/enh1`)
- Libri2mix (`egs2/librimix/enh1`)
- WHAMR! (`egs2/whamr/enh1`)
- DNS-Interspeech2020 dataset (`egs2/dns_ins20/enh1`)

In each `egs2` directory, you can find the pre-trained model under the `exp` directory.

One can easily use the pre-trained model to separate an audio mixture as follows:

```sh
# assuming you are now at ./egs2/wsj0_2mix/enh1
python separate.py \
    --model_path ./exp/enh_train_enh_tflocoformer_pretrained/valid.loss.ave_5best.pth \
    --audio_path /path/to/input_audio \
    --audio_output_dir /path/to/output_directory
```

## Example of training and inference

Here are example commands to run the WSJ0-2mix recipe.
Other dataset recipes are similar, but require additional steps (refer to the next section).

```sh
# Go to the corresponding example directory.
cd ../espnet/egs2/wsj0_2mix/enh1

# Data preparation and stats collection if necessary.
# NOTE: please fill the corresponding part of db.sh for data preparation.
./run.sh --stage 1 --stop_stage 5

# Training. We used 4 GPUs for training (batch size was 1 on each GPU; GPU RAM depends on dataset).
./run.sh --stage 6 --stop_stage 6 --enh_config conf/tuning/train_enh_tflocoformer.yaml --ngpu 4

# Inference.
./run.sh --stage 7 --stop_stage 7 --enh_config conf/tuning/train_enh_tflocoformer.yaml --ngpu 1 --gpu_inference true --inference_model valid.loss.ave_5best.pth

# Scoring. Scores are written in RESULT.md.
./run.sh --stage 8 --stop_stage 8 --enh_config conf/tuning/train_enh_tflocoformer.yaml
```

## Instructions for running training on each dataset in the ESPnet pipeline

Some recipe changes are required to run the experiments as in the paper.
After finishing the processes below, you can run the recipe in a normal way as described above.

### WHAMR!

First, please install pyroomacoustics: `pip install pyroomacoustics==0.2.0`.

The default task in ESPnet is noisy reverberant *speech enhancement without dereverberation* (using mix_single_reverb subset), while we did noisy reverberant *speech separation with dereverberation*.
To do the same task as in the paper, please run the following commands in `egs2/whamr/enh1`:

```sh
# Speech enhancement task -> speech separation task.
sed -i '13,15s|single|both|' run.sh
sed -i '23s|1|2|' run.sh

# Modify the url of the WHAM! noise and the WHAMR! script.
sed -i '42s|.*|  wham_noise_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip|' local/whamr_create_mixture.sh
sed -i '52s|.*|script_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/whamr_scripts.tar.gz|' local/whamr_create_mixture.sh

cd local && patch -b < whamr_data_prep.patch && cd ..
```

Then, you can start running the recipe from stage 1.

### Libri2mix

The default task in ESPnet is *noisy speech separation*, while we did *noise-free speech separation*.
To do the same task as in the paper, run the following commands in `egs2/librimix/enh1`:

```sh
# Apply the patch file to data.sh.
cd local && patch -b < data.patch && cd ..

# Use only train-360. By default, both train-100 and train-360 are used.
sed -i '12s|"train"|"train-360"|' run.sh

# Noisy separation -> noise-free separation.
sed -i '17s|true|false|' run.sh

# Data preparation in the "clean" condition (noise-free separation).
./run.sh --stage 1 --stop_stage 5 --local_data_opts "--sample_rate 8k --min_or_max min --cond clean"
```

### DNS interspeech2020 dataset

In the paper, we simulated 3000 hours of noisy speech: 2700 h for training and 300 h for validation.
To reproduce the paper's result, run the following commands in `egs2/dns_ins20/enh1`:

```sh
sed -i '18s|.*|total_hours=3000|' local/dns_create_mixture.sh
sed -i '19s|.*|snr_lower=-5|' local/dns_create_mixture.sh
sed -i '20s|.*|snr_upper=15|' local/dns_create_mixture.sh
```

We recommend reducing the size of the validation data to save training time since the validation loop with 300 h takes a very long time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and license

Released under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: Apache-2.0
```

The following patch files:

- `espnet2/tasks/enh.patch`
- `egs2/librimix/local/data.patch`
- `egs2/whamr/local/whamr_data_prep.patch`

include code from <https://github.com/espnet/espnet> (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2017 ESPnet Developers

SPDX-License-Identifier: Apache-2.0
SPDX-License-Identifier: Apache-2.0
```
