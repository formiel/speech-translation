# Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation

This is the codebase for the paper [*Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation*](https://arxiv.org/abs/2011.00747) (COLING 2020, Oral presentation). 

# News
- 02/11/2020: First release, with training recipes and pre-trained models.

# Table of Contents

1. [Pre-trained models](#1-pre-trained-models)
2. [Dependencies](#2-dependencies)
3. [Data preparation](#3-data-preparation)
4. [Training](#4-training)
5. [Decoding](#5-decoding)
6. [References](#6-references)


# 1. Pre-trained models
Pre-trained models are available for download in the links below. To replicate the results, please follow [Section 5 Decoding](#5-decoding).

|   |  type | side  | self  | src  | merge | epochs |  WER | BLEU | de | es | fr | it | nl | pt | ro | ru |
|---|:---|:---|:---|:---|:---|:---|:---|---|---|---|---|---|---|---|---|---|
|  <td colspan=5>Inaguma et al. [1]| 50 | 12.0| 25.05 | 22.91 | 27.96 | 32.69 | 23.75 | 27.43 | 28.01 | 21.90 | **15.75** |
|  <td colspan=5>Gangi et al. [2]|  | - | - | 17.70 | 20.90 | 26.50 | 18.00 | 20.00 | 22.60 | - | - |
|  <td colspan=5>Gangi et al. [2]|  | -| 17.55  | 16.50 | 18.90 | 24.50 | 16.20 | 17.80 | 20.80 | 15.90 | 9.80 |
|[Link](https://zenodo.org/record/4563217/files/dict1_bpe8k_independent_plus2.tar.gz?download=1)| `independent++` ||||| 25 | 11.6 | 24.60 | 22.82 |27.20 |32.11 |23.34 |26.67 |28.98 |21.37 |14.34 |
|[Link](https://zenodo.org/record/4563217/files/dict1_bpe8k_dual_decoder_xself_xsrc_xconcat.tar.gz?download=1)| `par` | `both` | :heavy_check_mark: | :heavy_check_mark: | `concat` | 25 | 11.6 | 25.00 | 22.74 |27.59 |32.86 |23.50 |26.97 |29.51 |21.94 |14.88 |    
|[Link](https://zenodo.org/record/4563217/files/dict1_bpe8k_dual_decoder_xsrc_xsum_waitk_asr3.tar.gz?download=1)| `par`<sup>`R3`</sup> | `both` | - | :heavy_check_mark: | `sum` | 25 | 11.6 | 24.87 |  22.84 |27.92 |32.12 |23.61 |27.29 |29.48 |21.16 |14.50 | 
|[Link](https://zenodo.org/record/4563217/files/dict1_bpe8k_dual_decoder_xsrc_xsum_plus2.tar.gz?download=1)| `par++` | `both` | - | :heavy_check_mark: | `sum` | 25 | **11.4**| **25.62** | **23.63** |**28.12** |**33.45** |**24.18** |**27.55** |**29.95** |**22.87** |15.21 |

[1] Inaguma et al., 2020. Espnet-st: All-in-one speech translation toolkit. (Bilingual one-to-one models)

[2] Gangi et al., 2019. One-to-many multilingual end-to-end speech translation.

# 2. Dependencies

You will need PyTorch, Kaldi, and ESPNet. **In the sequel, it is assumed that
you are already inside a virtual environment** with PyTorch installed (together with necessary standard
Python packages), and that `$WORK` is your working directory. 

**Note that the instructions here are different from the ones
on the official ESPNet repo (they install a miniconda virtual environment that
will be activated each time you run an ESPNet script)**.

## Kaldi

Clone the Kaldi repo:

```bash
cd $WORK
git clone https://github.com/kaldi-asr/kaldi.git
```

The following commands may require other dependencies, please install them accordingly.

Check and make its dependencies:

```bash
cd $WORK/kaldi/tools
bash extras/check_dependencies.sh
touch python/.use_default_python
make -j$(nproc)
```

Build Kaldi, replace the MKL paths with your system's ones:

```bash
cd $WORK/kaldi/src
./configure --shared \
    --use-cuda=no \
    --mkl-root=/some/path/linux/mkl \
    --mkl-libdir=/some/path/linux/mkl/lib/intel64_lin
make depend -j$(nproc)
make -j$(nproc)
```

**Important:** After installing Kaldi, make sure there's no `kaldi/tools/env.sh` and no `kaldi/tools/python/python`, otherwise there will be an error (`no module sentencepiece`) when running ESPNet.

## ESPNet

Clone this repo:

```bash
cd $WORK
git clone https://github.com/formiel/speech-translation.git
```

Prepare the dependencies:

```bash
cd $WORK/speech-translation
ln -s $WORK/kaldi tools/kaldi
pip install .
cd tools
git clone https://github.com/moses-smt/mosesdecoder.git moses
```

If you prefer to install it in editable mode, then replace the `pip install` line
with

```bash
pip install --user . && pip install --user -e .
```

<!-- ### Moses
```bash
cd tools
git clone https://github.com/moses-smt/mosesdecoder.git
``` -->


# 3. Data preparation
1. Run the following command to process features and prepare data in `json` format.

```bash
bash run.sh --ngpu 0 \
            --stage 0 \
            --stop-stage 2 \
            --must-c $MUSTC_ROOT \
            --tgt-langs de_es_fr_it_nl_pt_ro_ru \
            --nbpe 8000 \
            --nbpe_src 8000
```
where `MUSTC_ROOT` is directory where you save raw MuST-C data.

2. Create symlinks so that the processed data is saved in the required strutured for training.
```bash
python tools/create_symlinks.py --input-dir /path/to/dump \
                                --output-dir ${DATA_DIR} \
                                --use-lid --use-joint-dict
``` 
where `${DATA_DIR}` is the path to the data folder for training. Its structure is as below.
```
${DATA_DIR}
└──${tgt_langs}
    └──use_dict1
    |    └──src8000_tgt8000
    |    |   └──train_sp
    |    |        └──en-de.json
    |    |        └──en-es.json
    |    |        └──...
    |    |    └──dev
    |    |        └──en-de.json
    |    |        └──en-es.json
    |    |        └──...
    |    |    └──tst-COMMON
    |    |        └──en-de.json
    |    |        └──en-es.json
    |    |        └──...
    |    |    └──tst-HE
    |    |        └──en-de.json
    |    |        └──en-es.json
    |    |        └──...
    |    |    └──lang_1spm
    |    |        train_sp.en-${tgt_langs}_bpe8000_tc_${suffix}.txt
    |    └──src32000_tgt32000
    |        ....
    └──use_dict2
         └──src8000_tgt8000
         └──src8000_tgt32000
         ....
```
In which, `${tgt_langs}` is the target languages separated by `_`. For example, for a model trained on 8 languages, `${tgt_langs}` is `de_es_fr_it_nl_pt_ro_ru`.


# 4. Training
The training configurations are saved in `./conf/training`.

Please run the following command to train or resume training. The training will be automatically resumed from the last checkpoints in the `exp/${tag}/results` folder if this folder exists (and there are checkpoints of the format `snapshot.iter.${NUM_ITER}` in it), where `${tag}` is the name tag of the experiment and `${NUM_ITER}` is the iteration number. If `exp/${tag}/results` folder does not exist, the model will be trained from scratch (the weights is initialized using the pre-trained weights provided).

```bash
bash run.sh --stage 4 --stop-stage 4 --ngpu ${ngpu} \
            --preprocess-config ./conf/specaug.yaml \
            --datadir ${DATA_DIR} \
            --tgt-langs ${tgt_langs} \
            --tag ${tag}
```
where
- `${ngpu}`: number of GPUs to be used for training. Training on multi-node is currently not supported.
- `${DATA_DIR}`: path to the input folder (as described above).
- `${tag}`: name of the training configuration file (without `.yaml` extension).
- `${tgt_langs}`: the target languages separated by `_` (as described above).
<!-- - `${pretrained_weights}`: path to the pre-trained weights. -->

The checkpoints are saved in `./exp/${tag}/results`, and the tensorboard is saved in `./tensorboard/${tag}`. 


# 5. Decoding
The decoding configurations are saved in `./conf/decoding`.

Please run the following command for decoding.

```bash
bash run.sh --stage 5 --stop-stage 5 --ngpu 0 \
            --preprocess-config ./conf/specaug.yaml \
            --datadir ${DATA_DIR} \
            --tgt-langs ${tgt_langs} \
            --decode-config ${decode_config} \
            --trans-set ${trans_set} \
            --trans-model ${trans_model} \
            --tag ${tag}
```
where
- `${DATA_DIR}`, `${tgt_langs}`, and `${tag}` are same parameters as described above.
- `${trans_set}`: datasets to be decoded, seperated by space, e.g. `tst-COMMON.en-de.de tst-COMMON.en-fr.fr`, etc. If this value is an empty string, then the default datasets for decoding are `tst-COMMON` and `tst-HE` sets of all target languages.

<!-- The decoding results are saved under `./exp/${tag}/decode_${tag}_${decode_config}_${split}_${lang_pair}_${trans_model}`. -->


# 6. References
If you find the resources in this repository useful, please cite the following paper:
```
@inproceedings{le2020dualdecoder,
    title       = {Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation},
    author      = {Le, Hang and Pino, Juan and Wang, Changhan and Gu, Jiatao and Schwab, Didier and Besacier, Laurent},
    booktitle   = {Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020},
    publisher   = {Association for Computational Linguistics}
    year        = {2020}
}
```

This repo is a fork of [ESPNet](https://github.com/espnet/espnet). You should consider citing their papers as well if you use this code. 