# Multilingual Speech Translation

This is our fork of ESPNet. Code for multilingual speech translation can be found in `egs/must_c/st_multilingual`.

<!-- # Table of Contents

1. [Install necessary packages](#1.-install-necessary-packages) -->
<!-- 2. [Train multilingual models](#2.-train-multilingual-models)
3. [Decode multilingual models](#3.-decode-multilingual-models) -->


## 1. Dependencies

You will need PyTorch, Kaldi, and ESPNet. **In the sequel, it is assumed that
you are already inside a virtual environment** with PyTorch installed (together with necessary standard
Python packages), and that `$WORK` is your working directory. **Note that the instructions here are different from the ones
on the official ESPNet repo (they install a miniconda virtual environment that
will be activated each time you run an ESPNet script)**.

### Kaldi

Clone the Kaldi repo:

```bash
cd $WORK
git clone https://github.com/kaldi-asr/kaldi.git
```

The following commands may require other dependencies, please install them
accordingly.

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

**Important:** After installing Kaldi, make sure there's no `kaldi/tools/env.sh`
and no `kaldi/tools/python/python`, otherwise there will be an error
("no module sentencepiece") when running ESPNet.

### ESPNet

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


## 2. Instructions on training new models or resuming checkpoints

### 2.1. Download necessary files
All of the neccessary files can be downloaded from [here](). The folders `data`, `dump`, and `exp` should be placed under `egs/must_c/st_multilingual` as follows.
```
speech-translation
└──egs   
   └───must_c
       └───st_multilingual
            └───conf
            |    └───training
            |           config1.yaml
            |           config2.yaml
            └───data
            |   └───lang_1spm
            | 
            └───dump
            |   └───train_sp.en-de.de
            |   └───dev.en-de.de
            |   └───tst-COMMON.en-de.de
            |   └───tst-HE.en-de.de
            |   └───...
            |   
            └───exp
            |    └───config1
            |    |   |   train.log
            |    |   └───results
            |    |       └───snapshot.iter.5000
            |    |       └───snapshot.iter.10000
            |    |       └───...
            |    |           
            |    └───config2
            |    |   |   train.log
            |    |   └───results
            |    |       └───snapshot.iter.5000
            |    |       └───snapshot.iter.10000
            |    |       └───...
            |    └───... 
            |
            └───tensorboard
                └───config1
                └───config2
                └───...
```

#### a. Feature files
The features are saved in the `dump` folder. After saving this folder under `egs/must_c/st_multilingual`, please run the 2 steps below to prepare the data for training.

1. Mofidy the *hard-coded* feature paths in the json files
```bash
cd speech-translation/egs/must_c/st_multilingual

python modify_fpath.py --input-dir ./dump
```

2. Create symlinks so that the data is saved in the required strutured for training
```bash
python create_symlinks.py --output-dir ${DATA_DIR}
``` 
where `${DATA_DIR}` is the path to the input folder for training. Its structure is as below.
```
${DATA_DIR}
└──${tgt_langs}
    └──train_sp
        └──en-de.json
        └──en-es.json
        └──...
    └──dev
        └──en-de.json
        └──en-es.json
        └──...
    └──tst-COMMON
        └──en-de.json
        └──en-es.json
        └──...
    └──tst-HE
        └──en-de.json
        └──en-es.json
        └──...
    └──lang_1spm
            train_sp.en-${tgt_langs}.${tgt_langs}_bpe8000_units_tc_${suffix}.txt

```
In which, `${tgt_langs}` is the target languages separated by `_`. For example, for a model trained on 8 languages, `${tgt_langs}` is `de_es_fr_it_nl_pt_ro_ru`.

#### b. Dictionary files
Learned dictionaries are included in the folder `data/lang_1spm`.

#### c. Pre-trained weights
The pre-trained weights is saved in the folder `pre_trained_weights`.

#### d. Trained models
The trained models are saved in the folder `exp`.

### 2.2. List of configurations needed to be trained
The configurations needed to be trained for longer epochs are saved in `egs/must_c/st_multilingual/conf/training` in this repo.

### 2.3. Train or Resume training
Please run the following command to train or resume training. **The training will be automatically resumed from the last checkpoints in the `exp/${config}/results` folder if this folder exists, where `${config}` is the name tag of the experiment. If `exp/${config}/results` folder does not exist, the model will be trained from scratch (the weights is initialized using the pre-trained weights provided)**. 

```bash
bash run.sh --stage 4 --stop-stage 4 --ngpu 8 \
               --datadir ${DATA_DIR} \
               --preprocess-config ./conf/specaug.yaml \
               --tag ${tag} \
               --asr-model ${pretrained_weights} \
               --st-model ${pretrained_weights} \
               --tgt_lang ${tgt_langs}

```
where
- `${DATA_DIR}`: path to the input folder (as described above).
- `${tag}`: name of the configuration file (without `.yaml` extension).
- `${pretrained_weights}`: path to the pre-trained weights, which is located in `exp_data/pre_trained_weights/decoder_pre_1decoder/model.150k.acc.best` in the downloaded folder.
- `${tgt_langs}`: the target languages separated by `_` (as described above).
