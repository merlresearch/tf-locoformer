#!/usr/bin/env bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0

espnet_root=$1
cdir=${PWD}

# Copy the TF-Locoformer code.
cp ${cdir}/espnet2/enh/separator/tflocoformer_separator.py ${espnet_root}/espnet2/enh/separator

# Apply the patch file to enh.py to import TF-Locoformer.
# enh.py is directly modified by patching and the original enh.py is saved as enh.py.orig.
# .orig file can be deleted if it is not necessary.
cd ${espnet_root}/espnet2/tasks && patch -b < ${cdir}/espnet2/tasks/enh.patch && cd ${cdir}

# Copy other files.
for dset in wsj0_2mix whamr librimix dns_ins20; do
    # Copy separate.py on each egs2
    cp ${cdir}/egs2/wsj0_2mix/enh1/separate.py ${espnet_root}/egs2/${dset}/enh1

    # Copy the config file
    cp ${cdir}/egs2/${dset}/enh1/conf/tuning/train_enh_tflocoformer.yaml ${espnet_root}/egs2/${dset}/enh1/conf/tuning

    # Copy the pre-trained model
    mkdir -p ${espnet_root}/egs2/${dset}/enh1/exp
    cp -r ${cdir}/egs2/${dset}/enh1/exp/enh_train_enh_tflocoformer_raw ${espnet_root}/egs2/${dset}/enh1/exp/enh_train_enh_tflocoformer_pretrained

    # whamr has small and medium models
    if [ $dset = "whamr" ]; then
        cp ${cdir}/egs2/${dset}/enh1/conf/tuning/train_enh_tflocoformer_small.yaml ${espnet_root}/egs2/${dset}/enh1/conf/tuning
        cp -r ${cdir}/egs2/${dset}/enh1/exp/enh_train_enh_tflocoformer_small_raw ${espnet_root}/egs2/${dset}/enh1/exp/enh_train_enh_tflocoformer_small_pretrained
    fi
done

# Copy patch files.
cp ${cdir}/egs2/whamr/enh1/local/whamr_data_prep.patch ${espnet_root}/egs2/whamr/enh1/local
cp ${cdir}/egs2/librimix/enh1/local/data.patch ${espnet_root}/egs2/librimix/enh1/local
