#!/usr/bin/env python3
# encoding: utf-8

"""
Modified by Hang Le
The original copyright is appended below
--
Copyright 2019 Kyoto University (Hirofumi Inaguma)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

"""Training/decoding definition for joint ASR and multilingual task."""

import copy
import json
import logging
import multiprocessing
import numpy as np
import os
import random
import six
import sys
import shutil
import time
import torch
import warnings

from collections import deque
from tensorboardX import SummaryWriter

from chainer import training, reporter, utils
from chainer import serializer as serializer_module
from chainer.training import extensions, extension, util
from chainer.training import trigger as trigger_module
from chainer.utils import argument

from espnet.asr.asr_utils import (
    adadelta_eps_decay, adam_lr_decay,
    add_results_to_json, add_results_to_json_st_asr,
    CompareValueTrigger,
    get_model_conf,
    restore_snapshot, snapshot_object,
    torch_load, torch_resume, torch_snapshot
) 
from espnet.asr.pytorch_backend.asr_init import (
    load_trained_model, load_trained_model_multi,
    load_trained_modules, load_trained_modules_multi
) 

from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.st_interface import STInterface
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop, set_early_stop

from espnet.asr.pytorch_backend.asr import (
    CustomEvaluator, CustomUpdater,
    CustomConverter as AsrCustomConverter
)

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


class BestValueTrigger(object):

    """Trigger invoked when specific value becomes best.

    Args:
        key (str): Key of value.
        compare (callable): Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, compare, trigger=(1, 'epoch'), 
                best_value=None, verbose=True):
        self._key = key
        self._best_value = best_value
        self._interval_trigger = util.get_trigger(trigger)
        self._init_summary()
        self._compare = compare
        self.verbose = verbose

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.

        Returns:
            bool: ``True`` if the corresponding extension should be invoked in
            this iteration.

        """

        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._best_value is None or self._compare(self._best_value, value):
            if self.verbose:
                logging.info(f'{self._key} improved from {self._best_value} to {value}')
            self._best_value = value
            return True
        return False

    def _init_summary(self):
        self._summary = reporter.DictSummary()

    def serialize(self, serializer):
        self._interval_trigger.serialize(serializer['interval_trigger'])
        self._summary.serialize(serializer['summary'])
        self._best_value = serializer('best_value', self._best_value)


class MaxValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes maximum.

    For example you can use this trigger to take snapshot on the epoch the
    validation accuracy is maximum.

    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes maximum.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, trigger=(1, 'epoch'), best_value=None, verbose=True):
        super(MaxValueTrigger, self).__init__(
            key, lambda max_value, new_value: new_value > max_value, trigger,
            best_value=best_value, verbose=verbose)


class MinValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes minimum.

    For example you can use this trigger to take snapshot on the epoch the
    validation loss is minimum.

    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes minimum.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, trigger=(1, 'epoch'), best_value=None, verbose=True):
        super(MinValueTrigger, self).__init__(
            key, lambda min_value, new_value: new_value < min_value, trigger,
            best_value=best_value, verbose=verbose)


class TimeLimitTrigger(object):
    """Trigger for time limit
    """

    def __init__(self, args):
        self._max_trigger = util.get_trigger((args.epochs, 'epoch'))
        key_trigger = 'iteration' if args.save_interval_iters > 0 else 'epoch'
        self._interval_trigger = util.get_trigger((args.save_interval_iters, key_trigger))
        self.start_time = args.start_time
        self.time_limit = args.time_limit * 60
        self.max_epochs = args.epochs

    def __call__(self, trainer):
        """Decides whether the training loop should be stopped.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with.

        Returns:
            bool: ``True`` if the training loop should be stopped.
        """
        if self._max_trigger(trainer):
            logging.info(f'Training has reached specified number of epochs: {self.max_epochs}.')
            return True
        
        if self._interval_trigger(trainer):
            if not hasattr(trainer, 'elapsed_times'):
                trainer.elapsed_times = deque(maxlen=6)
            trainer.elapsed_times.append(trainer.elapsed_time)
            l = len(trainer.elapsed_times)
            if l > 1:
                A = np.zeros((l-1, l))
                for i in range(l-1):
                    A[i, i:i+2] = [-1, 1]        
                trainer.snapshot_elapsed_times = np.matmul(A, trainer.elapsed_times)
            else:
                trainer.snapshot_elapsed_times = np.zeros(1)

            est_elapsed_time = (time.time() - self.start_time + \
                                np.mean(trainer.snapshot_elapsed_times)) / 60
            logging.info(f'Estimated next epoch elapsed time {est_elapsed_time:.2f}min. \
                            Time limit: {self.time_limit:.2f}min.')
            if est_elapsed_time < self.time_limit:
                logging.info('Continue training...')
                return False
            else:
                logging.info('Time limit reached. Stop training.')
                return True

    def get_training_length(self):
        return self._max_trigger.get_training_length()


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def shuffle_batches(batches_list, shuffle=True):
    if shuffle:
        indices = [i for i in range(len(batches_list))]
        random.shuffle(indices)
        return [batches_list[i] for i in indices]
    else:
        return batches_list
    

class StAsrCustomConverter(AsrCustomConverter):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.
        asr_task (bool): multi-task with ASR task.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32, asr_task=False):
        """Construct a CustomConverter object."""
        super().__init__(subsampling_factor=subsampling_factor, dtype=dtype)
        self.asr_task = asr_task

    def __call__(self, batch, device=torch.device('cpu')):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        _, ys = batch[0]
        ys_asr = copy.deepcopy(ys)
        xs_pad, ilens, ys_pad = super().__call__(batch, device)
        if self.asr_task:
            ys_pad_asr = pad_list([
                torch.from_numpy(
                    np.insert(y[1][1], 0, y[1][0]) if isinstance(y[1], tuple) 
                    else np.array(y[1][:]) if isinstance(y, tuple)
                    else y).long()
                for y in ys_asr], self.ignore_id).to(device)
        else:
            ys_pad_asr = None

        return xs_pad, ilens, ys_pad, ys_pad_asr


class MtCustomConverter(object):
    """Custom batch converter for Pytorch."""

    def __init__(self):
        """Construct a CustomConverter object."""
        self.ignore_id = -1
        self.pad = 0
        # NOTE: we reserve index:0 for <pad> although this is reserved for a blank class
        # in ASR. However, blank labels are not used in NMT. To keep the vocabulary size,
        # we use index:0 for padding instead of adding one more class.

    def __call__(self, batch, device=torch.device('cpu')):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        ys_asr, ys = batch[0]

        # get batch of lengths of input sequences
        ilens = np.array([y[1].shape[0] if isinstance(y, tuple) else y.shape[0]
                        for y in ys_asr])
        ilens = torch.from_numpy(ilens).to(device)

        # perform padding and convert to tensor
        ys_pad = pad_list([
            torch.from_numpy(
                np.insert(y[1], 0, y[0]) if isinstance(y, tuple) else y).long()
            for y in ys], self.ignore_id).to(device)

        ys_pad_asr = pad_list([
            torch.from_numpy(
                np.insert(y[1], 0, y[0]) if isinstance(y, tuple) else y).long()
            for y in ys_asr], self.pad).to(device)

        return None, ilens, ys_pad, ys_pad_asr


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # language pairs
    # removed_langs = getattr(args, "removed_langs", None)
    # lang_pairs_dict = getattr(args, "lang_pairs_dict")
    # if removed_langs is not None:
    #     removed_langs = removed_langs.split("_")
    #     removed_langs = [f"en-{l}" for l in removed_langs]
    #     lang_pairs_dict = lang_pairs_dict.split(",")
    #     lang_pairs_dict = ",".join(list(set(lang_pairs_dict) - set(removed_langs)))
    # lang_pairs = args.lang_pairs if (not args.train_adapters and 
    #                             not args.use_multi_dict and not removed_langs) \
    #                             else lang_pairs_dict
    lang_pairs = sorted(args.lang_pairs.split(','))
    args.one_to_many = True if len(lang_pairs) > 1 else False
    if args.one_to_many:
        assert args.use_lid
    tgt_langs = sorted([p.split('-')[-1] for p in lang_pairs])
    src_lang = lang_pairs[0].split('-')[0]
    args.src_lang = src_lang
    
    # get paths to data
    if args.one_to_many and not args.use_multi_dict:
        train_jpaths = [os.path.join(args.train_json, fname) for fname in 
                sorted(os.listdir(args.train_json)) if fname.endswith('.json')
                and fname.split(".")[0] in args.lang_pairs.split(",")]
        valid_jpaths = [os.path.join(args.valid_json, fname) for fname in 
                sorted(os.listdir(args.valid_json)) if fname.endswith('.json')
                and fname.split(".")[0] in args.lang_pairs.split(",")]
    else:
        train_jpaths = [args.train_json]
        valid_jpaths = [args.valid_json]   

    # prepare language ID tokens for one-to-many systems 
    if args.use_lid:
        if args.use_joint_dict:
            all_langs = list(sorted(set([l for p in lang_pairs for l in p.split('-')])))
            langs_dict = {}
            offset = 2 # for <blank> and <unk>
            for i, lang in enumerate(all_langs):
                langs_dict[f'<2{lang}>'] = offset + i
            args.langs_dict_tgt = langs_dict
            args.langs_dict_src = langs_dict
        else:
            langs_dict_tgt = {}
            langs_dict_src = {}
            offset = 2 # for <blank> and <unk>
            for i, lang in enumerate(tgt_langs):
                langs_dict_tgt[f'<2{lang}>'] = offset + i
            langs_dict_src[f'<2{src_lang}>'] = offset
            args.langs_dict_tgt = langs_dict_tgt
            args.langs_dict_src = langs_dict_src
    else:
        args.langs_dict_tgt = None
        args.langs_dict_src = None

    # get input and output dimension info 
    idims = []
    odims_src = []
    odims_tgt = []
    for i, jpath in enumerate(valid_jpaths):
        with open(jpath, 'rb') as f:
            valid_json = json.load(f)['utts']
        utts = list(valid_json.keys())
        idims.append(int(valid_json[utts[0]]['input'][0]['shape'][-1]))
        odims_tgt.append(int(valid_json[utts[0]]['output'][0]['shape'][-1]))
        odims_src.append(int(valid_json[utts[0]]['output'][1]['shape'][-1]))

    assert len(set(idims)) == 1; idim = idims[0]
    assert len(set(odims_tgt)) == 1; odim_tgt = odims_tgt[0]
    assert len(set(odims_src)) == 1; odim_src = odims_src[0]
    logging.info('#input dims : ' + str(idim))
    logging.info('#tgt output dims: ' + str(odim_tgt))
    logging.info('#src output dims: ' + str(odim_src))

    # adapters
    if args.train_adapters:
        args.use_adapters = True
    if args.use_adapters or args.use_multi_dict:
        lang_pairs = sorted(args.lang_pairs.split(',')) # reset lang_pairs
        tgt_langs = sorted([p.split('-')[-1] for p in lang_pairs])
        all_langs = list(sorted(set([l for p in lang_pairs for l in p.split('-')]))) \
                    if args.asr_weight > 0.0 else tgt_langs
        if args.use_adapters:
            args.adapters = [src_lang] if args.asr_weight > 0.0 and not args.do_st \
                else [l for l in all_langs] if (args.use_adapters_for_asr or args.use_adapters_in_enc) \
                else [l for l in tgt_langs]
        else:
            args.adapters = None
    else:
        args.adapters = None
    if args.use_adapters:
        args.homogeneous_batch = True

    # Initialize with pre-trained ASR encoder and MT decoder
    if args.enc_init is not None or args.dec_init is not None:
        logging.info('Loading pretrained encoder and/or decoder ...')
        if args.enc_init is not None:
            assert os.path.isfile(args.enc_init)
        if args.dec_init is not None:
            assert os.path.isfile(args.dec_init)
        model = load_trained_modules_multi(idim, odim_tgt, odim_src, args, 
                                            interface=STInterface)
        logging.info(f'*** Model *** \n {model}')
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim_tgt, odim_src, args)
        logging.info(f'*** Model *** \n {model}')
    assert isinstance(model, STInterface)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f'| Number of parameters: {num_params}')

    subsampling_factor = model.subsample[0]

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    args.do_mt = getattr(args, "mt_weight", 0.0) > 0.0
    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim_tgt, odim_src, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        if "char_list" not in key:
            logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))
        else:
            logging.info(f'ARGS: {key}: {vars(args)[key][:10]}, ..., {vars(args)[key][-5:]}')

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32

    # freeze all weights except for trainable modules in adapters or when do_ft
    if (args.use_adapters and not args.train_adapters) or args.do_ft:
        trainable_modules = args.trainable_modules.split(",")
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            for m in trainable_modules:
                if m in n:
                    p.requires_grad = True
        logging.info('***** Trainable parameters *****')
        for n, p in model.named_parameters():
            if p.requires_grad:
                logging.info(f'- {n}: {p.is_leaf}')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_params}')

    model = model.to(device=device, dtype=dtype)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(f"You need to install apex for --train-dtype {args.train_dtype}. "
                          "See https://github.com/NVIDIA/apex#linux")
            raise e
        if args.opt == 'noam':
            model, optimizer.optimizer = amp.initialize(model, optimizer.optimizer, 
                                                    opt_level=args.train_dtype)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.train_dtype)
        use_apex = True
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))
    
    # read json data
    num_langs = len(tgt_langs)
    train_all_pairs = [None] * num_langs
    valid_all_pairs = [None] * num_langs
    batch_size = args.batch_size//num_langs if (num_langs > 1 and 
                                                (args.do_st or args.do_mt) and
                                                not args.homogeneous_batch) \
                                            else args.batch_size
    min_batch_size = args.ngpu if (args.ngpu > 1 and args.do_mt) else 1

    for i, jpath in enumerate(train_jpaths):
        with open(jpath, 'rb') as f:
            train_json = json.load(f)['utts']
            train_all_pairs[i] = make_batchset(train_json, batch_size,
                            args.maxlen_in, args.maxlen_out, args.minibatches,
                            min_batch_size=min_batch_size,
                            shortest_first=use_sortagrad,
                            count=args.batch_count,
                            batch_bins=args.batch_bins,
                            batch_frames_in=args.batch_frames_in,
                            batch_frames_out=args.batch_frames_out,
                            batch_frames_inout=args.batch_frames_inout,
                            mt=args.do_mt, iaxis=1 if args.do_mt else 0)         
    for i, jpath in enumerate(valid_jpaths):
        with open(jpath, 'rb') as f:
            valid_json = json.load(f)['utts']
            valid_all_pairs[i] = make_batchset(valid_json, batch_size,
                            args.maxlen_in, args.maxlen_out, args.minibatches,
                            min_batch_size=min_batch_size,
                            count=args.batch_count,
                            batch_bins=args.batch_bins,
                            batch_frames_in=args.batch_frames_in,
                            batch_frames_out=args.batch_frames_out,
                            batch_frames_inout=args.batch_frames_inout,
                            mt=args.do_mt, iaxis=1 if args.do_mt else 0)

    if num_langs > 1:
        cycle_train = [cycle(x) for x in train_all_pairs]
        cycle_valid = [cycle(x) for x in valid_all_pairs]
        num_batches_train = max(len(i) for i in train_all_pairs)
        num_batches_valid = max(len(i) for i in valid_all_pairs)
        if (args.do_st or args.do_mt) and not args.homogeneous_batch:
            cycle_train = [cycle(x) for x in train_all_pairs]
            cycle_valid = [cycle(x) for x in valid_all_pairs]

            num_batches_train = max(len(i) for i in train_all_pairs)
            num_batches_valid = max(len(i) for i in valid_all_pairs)
            train = [None] * num_batches_train
            valid = [None] * num_batches_valid

            for i, s in enumerate(zip(*cycle_train)):
                x = []
                for y in s:
                    x.extend(y)
                train[i] = x
                if i >= num_batches_train - 1:
                    break
            for i, s in enumerate(zip(*cycle_valid)):
                x = []
                for y in s:
                    x.extend(y)
                valid[i] = x
                if i >= num_batches_valid - 1:
                    break
        else:
            train = []
            valid = []
            for i, s in enumerate(zip(*cycle_train)):
                for y in s:
                    train.append(y)
                if i >= num_batches_train - 1:
                    break
            for i, s in enumerate(zip(*cycle_valid)):
                for y in s:
                    valid.append(y)
                if i >= num_batches_valid - 1:
                    break
    else:
        train = train_all_pairs[0]
        valid = valid_all_pairs[0]
    
    # n = len(train)
    # n_samples = 0
    # for i, batch in enumerate(train):
    #     logging.info(f'batch {i}/{n}')
    #     m = len(batch)
    #     n_tokens = 0
    #     if i == 0:
    #         for j, sample in enumerate(batch):
    #             logging.info(f'sample: {sample}')
    #         break
    #         logging.info(f'sample {j}/{m}: {sample[1]["lang"]}, \
    #                     {sample[1]["input"][0]["shape"][0]}, \
    #                     {sample[1]["input"][1]["shape"][0]}')
    #         n_samples += 1
    #         n_tokens += sample[1]["input"][0]["shape"][0]
    #     logging.info(f'Total number of tokens in batch: {n_tokens}')
    # logging.info(f'Total number of training samples: {n_samples}')

    load_tr = LoadInputsAndTargets(mode='mt' if args.do_mt else 'asr',
            load_input=not args.do_mt,
            preprocess_conf=args.preprocess_conf if not args.do_mt else None,
            preprocess_args={'train': True},
            langs_dict_tgt=args.langs_dict_tgt,
            langs_dict_src=args.langs_dict_src,
            src_lang=src_lang,
            lang_tok_mt=args.lang_tok_mt)
    load_cv = LoadInputsAndTargets(mode='mt' if args.do_mt else 'asr',
            load_input=not args.do_mt,
            preprocess_conf=args.preprocess_conf if not args.do_mt else None,
            preprocess_args={'train': False},
            langs_dict_tgt=args.langs_dict_tgt,
            langs_dict_src=args.langs_dict_src,
            src_lang=src_lang,
            lang_tok_mt=args.lang_tok_mt)

    # features, targets = load_tr(train[0])
    # logging.info(f'features \n {features}')
    # logging.info(f'targets \n {targets}')

    # Setup a converter
    if args.do_mt:
        converter = MtCustomConverter()
    else:
        converter = StAsrCustomConverter(subsampling_factor=subsampling_factor, 
                                        dtype=dtype,
                                        asr_task=args.asr_weight > 0)
    # xs_pad, ilens, ys_pad, ys_pad_asr = converter([load_tr(train[0])])
    # logging.info(f'xs_pad: {xs_pad}')
    # logging.info(f'ilens: {ilens}')
    # logging.info(f'ys_pad: {ys_pad}')
    # logging.info(f'ys_pad_asr: {ys_pad_asr}')

    # hack to make batchsize argument as 1
    # actual batchsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    n_iter_processes = args.n_iter_processes
    if n_iter_processes < 0:
        n_iter_processes = multiprocessing.cpu_count()
    elif n_iter_processes > 0:
        n_iter_processes = min(n_iter_processes, multiprocessing.cpu_count())
    
    train_iter = {'main': ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)])),
        batch_size=1, num_workers=n_iter_processes,
        shuffle=not use_sortagrad, collate_fn=lambda x: x[0], 
        pin_memory=False)}
    valid_iter = {'main': ChainerDataLoader(
        dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
        batch_size=1, shuffle=False, collate_fn=lambda x: x[0],
        num_workers=n_iter_processes, 
        pin_memory=False)}

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer,
        device, args.ngpu, args.grad_noise, args.accum_grad, use_apex=use_apex)

    time_limit_trigger = TimeLimitTrigger(args)
    trainer = training.Trainer(
        updater, time_limit_trigger, out=args.outdir)

    if use_sortagrad:
        logging.info(f'use_sortagrad ...')
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Evaluate the model with the test dataset for each epoch
    if args.save_interval_iters > 0:
        trainer.extend(CustomEvaluator(model, valid_iter, reporter, device, args.ngpu),
                       trigger=(args.save_interval_iters, 'iteration'))
    else:
        trainer.extend(CustomEvaluator(model, valid_iter, reporter, device, args.ngpu))

    # Save attention weight each epoch
    if args.num_save_attention > 0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, transform=load_cv, device=device)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None

    # Make a plot for training and validation values
    if args.do_plots:
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                            'main/loss_asr', 'validation/main/loss_asr',
                                            'main/loss_st', 'validation/main/loss_st'],
                                            'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc',
                                            'main/acc_asr', 'validation/main/acc_asr'],
                                            'epoch', file_name='acc.png'))
        trainer.extend(extensions.PlotReport(['main/bleu', 'validation/main/bleu'],
                                            'epoch', file_name='bleu.png'))
        trainer.extend(extensions.PlotReport(['main/ppl', 'validation/main/ppl'],
                                            'epoch', file_name='ppl.png'))

    # Save best models
    key_to_save = 'validation/main/acc' if (args.do_st or args.do_mt) \
                                        else 'validation/main/acc_asr'
    if args.report_interval_iters > 0:
        trainer.extend(
            snapshot_object(model, 'model.loss.best'),
            trigger=MinValueTrigger('validation/main/loss',
                                    trigger=(args.report_interval_iters, 'iteration'),
                                    best_value=None))
        trainer.extend(
            snapshot_object(model, 'model.acc.best'),
            trigger=MaxValueTrigger(key_to_save,
                                    trigger=(args.report_interval_iters, 'iteration'),
                                    best_value=None))
    else:
        trainer.extend(
            snapshot_object(model, 'model.loss.best'),
            trigger=MinValueTrigger('validation/main/loss', best_value=None))
        trainer.extend(
            snapshot_object(model, 'model.acc.best'),
            trigger=MaxValueTrigger(key_to_save, best_value=None))

    # save snapshot which contains model and optimizer states
    if args.save_interval_iters > 0:
        trainer.extend(torch_snapshot(filename='snapshot.iter.{.updater.iteration}'),
                       trigger=(args.save_interval_iters, 'iteration'))
    else:
        trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if 'acc' in args.criterion:
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               key_to_save,
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               key_to_save,
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
    elif args.opt == 'adam':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               key_to_save,
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adam_lr_decay(args.lr_decay),
                           trigger=CompareValueTrigger(
                               key_to_save,
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adam_lr_decay(args.lr_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, 'iteration')))
    report_keys = ['epoch', 'iteration', 
                    'main/loss', 'validation/main/loss',  
                    'main/acc', 'validation/main/acc',
                    'elapsed_time']
    if args.do_st:
        report_keys.append('main/loss_st')
        report_keys.append('validation/main/loss_st')
    if args.asr_weight > 0:
        report_keys.append('main/acc_asr')
        report_keys.append('validation/main/acc_asr')
        report_keys.append('main/loss_asr')
        report_keys.append('validation/main/loss_asr')
    if args.do_mt:
        report_keys.append('main/ppl')
        report_keys.append('validation/main/ppl')

    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('eps')
    elif args.opt in ['adam', 'noam']:
        trainer.extend(extensions.observe_value(
            'lr', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["lr"]),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('lr')
    if args.asr_weight > 0:
        if args.mtlalpha > 0:
            report_keys.append('main/cer_ctc')
            report_keys.append('validation/main/cer_ctc')
        if args.mtlalpha < 1:
            if args.report_cer:
                report_keys.append('validation/main/cer')
            if args.report_wer:
                report_keys.append('validation/main/wer')
    if args.report_bleu:
        report_keys.append('validation/main/bleu')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(args.report_interval_iters, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(TensorboardLogger(SummaryWriter(args.tensorboard_dir), att_reporter),
                       trigger=(args.report_interval_iters, "iteration"))
    
    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def trans(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    logging.info(f'eval-no-adapters: {args.eval_no_adapters}')
    logging.info(f'adapter-path: {args.adapter_path}')
    model, train_args = load_trained_model_multi(args.model, 
                                        eval_no_adapters=args.eval_no_adapters,
                                        adapter_path=args.adapter_path)
    assert isinstance(model, STInterface)
    # args.ctc_weight = 0.0
    model.trans_args = args

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError("use '--api v2' option to decode with non-default language model")
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list_tgt), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.trans_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    train_args.do_mt = getattr(train_args, 
                              "do_mt", 
                              getattr(train_args, "mt_weight", 0.0) > 0.0)
    if not train_args.do_mt:
        load_inputs_and_targets = LoadInputsAndTargets(
            mode='asr', load_output=False, sort_in_input_length=False,
            preprocess_conf=train_args.preprocess_conf
            if args.preprocess_conf is None else args.preprocess_conf,
            preprocess_args={'train': False})

    # Change to evaluation mode
    model.eval()

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('| (%d/%d) decoding ' + name, idx, len(js.keys()))
                if train_args.do_mt:
                    feat = [js[name]['output'][1]['tokenid'].split()]
                else:
                    batch = [(name, js[name])]
                    feat = load_inputs_and_targets(batch)[0][0]

                if args.recog_and_trans: # for dual decoders
                    logging.info('***** Recognize and Translate simultaneously ******')
                    if args.beam_search_type == 'sum':
                        logging.info('=== Beam search by sum of scores ===')
                        nbest_hyps = model.recognize_and_translate_sum(
                                    feat, args, 
                                    train_args.char_list_tgt,
                                    train_args.char_list_src, 
                                    rnnlm, 
                                    decode_asr_weight=args.decode_asr_weight,
                                    score_is_prob=args.score_is_prob,
                                    ratio_diverse_st=args.ratio_diverse_st,
                                    ratio_diverse_asr=args.ratio_diverse_asr,
                                    use_rev_triu_width=args.use_rev_triu_width,
                                    use_diag=args.use_diag,
                                    debug=args.debug)
                        new_js[name] = add_results_to_json_st_asr(
                                                    js[name], nbest_hyps, 
                                                    train_args.char_list_tgt,
                                                    train_args.char_list_src)
                    elif args.beam_search_type == 'half-joint':
                        nbest_hyps = model.recognize_and_translate_half_joint(
                                                    feat, args, 
                                                    train_args.char_list_tgt,
                                                    train_args.char_list_src, 
                                                    rnnlm)
                        new_js[name] = add_results_to_json_st_asr(
                                                    js[name], nbest_hyps, 
                                                    train_args.char_list_tgt,
                                                    train_args.char_list_src)
                    elif args.beam_search_type == 'separate':
                        logging.info('=== Beam search using beam_cross hypothesis ===')
                        nbest_hyps, nbest_hyps_asr = model.recognize_and_translate_separate(
                                                    feat, args, 
                                                    train_args.char_list_tgt,
                                                    train_args.char_list_src,
                                                    rnnlm)
                        new_js[name] = add_results_to_json(js[name], 
                                                           nbest_hyps, 
                                                           train_args.char_list_tgt)
                        new_js[name]['output'].append(add_results_to_json(
                                                        js[name], nbest_hyps_asr, 
                                                        train_args.char_list_src, 
                                                        output_idx=1)['output'][0])
                    else:
                        raise NotImplementedError
                elif args.recog and args.trans:
                    logging.info('***** Recognize and Translate separately ******')
                    nbest_hyps_asr = model.recognize(feat, args, 
                                                     train_args.char_list_src, 
                                                     rnnlm)
                    nbest_hyps = model.translate(feat, args, 
                                                 train_args.char_list_tgt, 
                                                 rnnlm)
                    new_js[name] = add_results_to_json(js[name], 
                                                       nbest_hyps, 
                                                       train_args.char_list_tgt)
                    new_js[name]['output'].append(
                        add_results_to_json(js[name], 
                                            nbest_hyps_asr, 
                                            train_args.char_list_src, 
                                            output_idx=1)['output'][0])
                elif args.recog:
                    logging.info('***** Recognize only ******')
                    nbest_hyps_asr = model.recognize(feat, args, 
                                                     train_args.char_list_src, 
                                                     rnnlm)
                    new_js[name] = add_results_to_json(js[name], 
                                                       nbest_hyps_asr, 
                                                       train_args.char_list_src, 
                                                       output_idx=1)
                elif args.trans:
                    logging.info('***** Translate only ******')
                    nbest_hyps = model.translate(feat, args, 
                                                 train_args.char_list_tgt, 
                                                 rnnlm)
                    new_js[name] = add_results_to_json(js[name], 
                                                       nbest_hyps, 
                                                       train_args.char_list_tgt, 
                                                       output_idx=0)
                else:
                    raise NotImplementedError

    else:
        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = load_inputs_and_targets(batch)[0]
                nbest_hyps = model.translate_batch(feats, args, train_args.char_list, rnnlm=rnnlm)

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))