#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech translation model decoding script."""

import configargparse
import logging
import os
import random
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool

# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description='Translate text from speech using a speech translation model on one CPU or GPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add('--config', is_config_file=True,
               help='Config file path')
    parser.add('--config2', is_config_file=True,
               help='Second config file path that overwrites the settings in `--config`')
    parser.add('--config3', is_config_file=True,
               help='Third config file path that overwrites the settings in `--config` and `--config2`')

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of GPUs')
    parser.add_argument('--dtype', choices=("float16", "float32", "float64"), default="float32",
                        help='Float precision (only available in --api v2)')
    parser.add_argument('--backend', type=str, default='chainer',
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--debugmode', type=int, default=1,
                        help='Debugmode')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', type=int, default=1,
                        help='Verbose option')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='Batch size for beam search (0: means no batch processing)')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    parser.add_argument('--api', default="v1", choices=["v1", "v2"],
                        help='''Beam search APIs
        v1: Default API. It only supports the ASRInterface.recognize method and DefaultRNNLM.
        v2: Experimental API. It supports any models that implements ScorerInterface.''')
    # task related
    parser.add_argument('--trans-json', type=str,
                        help='Filename of translation data (json)')
    parser.add_argument('--result-label', type=str, required=True,
                        help='Filename of result label data (json)')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    # search related
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', type=float, default=0.0,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', type=float, default=0.0,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    
    parser.add_argument('--beam-size-asr', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty-asr', type=float, default=0.0,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio-asr', type=float, default=0.0,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio-asr', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                        help='')
    parser.add_argument('--recog', default=False, type=strtobool)
    parser.add_argument('--trans', default=False, type=strtobool)
    parser.add_argument('--debug', default=False, type=strtobool)
    parser.add_argument('--recog-and-trans', default=False, type=strtobool)
    parser.add_argument('--use-rev-triu-width', default=0, type=int)
    parser.add_argument('--use-diag', default=False, type=strtobool)
    parser.add_argument('--beam-search-type', type=str, choices=['sum', 'separate', 'sum-mono', 'half-joint'], default='separate',
                        help='Beam search type when doing recognition and translation simultaneously.')
    parser.add_argument('--beam-cross-size', type=int, default=1,
                        help='Beam cross size')
    parser.add_argument('--decode-asr-weight', type=float, default=1.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--score-is-prob', default=False, type=strtobool,
                        help='Score is probability or not when decoding.')
    parser.add_argument('--ratio-diverse-st', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ratio-diverse-asr', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--wait-k-asr', type=int, default=0,
                        help='ASR waits ST for k steps.')
    parser.add_argument('--wait-k-st', type=int, default=0,
                        help='ST waits ASR for k steps.')
    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--word-rnnlm', type=str, default=None,
                        help='Word RNNLM model file to read')
    parser.add_argument('--word-rnnlm-conf', type=str, default=None,
                        help='Word RNNLM model config file to read')
    parser.add_argument('--word-dict', type=str, default=None,
                        help='Word list to read')
    parser.add_argument('--lm-weight', type=float, default=0.1,
                        help='RNNLM weight')
    # multilingual related
    parser.add_argument('--tgt-lang', default=False, type=str,
                        help='target language ID (e.g., <en>, <de>, and <fr> etc.)')
    # adapter related
    parser.add_argument('--eval-no-adapters', default=False, type=strtobool,
                        help='Evaluate without adapters in the model.')
    parser.add_argument('--adapter-path', type=str, default='',
                        help='Pre-trained adapter path to be evaluated.')
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # validate rnn options
    if args.rnnlm is not None and args.word_rnnlm is not None:
        logging.error("It seems that both --rnnlm and --word-rnnlm are specified. Please use either option.")
        sys.exit(1)

    # trans
    logging.info('backend = ' + args.backend)
    if args.backend == "pytorch":
        # Experimental API that supports custom LMs
        from espnet.st.pytorch_backend.st import trans
        if args.dtype != "float32":
            raise NotImplementedError(f"`--dtype {args.dtype}` is only available with `--api v2`")
        trans(args)
    else:
        raise ValueError("Only pytorch are supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
