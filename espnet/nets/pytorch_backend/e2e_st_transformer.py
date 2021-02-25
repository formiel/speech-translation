"""
Modified by Hang Le
The original copyright is appended below
--
Copyright 2019 Kyoto University (Hirofumi Inaguma)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

"""Transformer model for joint ASR and multilingual ST"""

import logging
import math
import numpy as np
import six
import time

from argparse import Namespace
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_st import Reporter
from espnet.nets.pytorch_backend.e2e_mt import Reporter as MTReporter
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_pad_mask, pad_list,
    th_accuracy,
    to_device
) 

from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import (
    subsequent_mask, create_cross_mask, target_mask
) 
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.st_interface import STInterface
from espnet.nets.e2e_asr_common import end_detect


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
    

def build_embedding(dictionary, embed_dim, padding_idx=0):
    num_embeddings = max(list(dictionary.values())) + 1
    emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
    return emb    


class E2E(STInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', 
                            default=None, type=float,
                            help='dropout in transformer attention. \
                               Use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', 
                            default=True, type=strtobool,
                           help='normalize loss by length')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers \
                            (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        # Adapters
        group.add_argument('--adapter-reduction-factor', default=None, type=float,
                           help='Reduction factor in bottle neck of adapter modules for decoder')
        group.add_argument('--adapter-reduction-factor-enc', default=None, type=float,
                           help='Reduction factor in bottle neck of adapter modules for encoder')
        group.add_argument('--adapter-before-src-attn', default=False, type=strtobool,
                           help='Add adapter before src attn module in decoder')
        group.add_argument('--adapter-after-mha', default=False, type=strtobool,
                           help='Add adapter after multi-head attention')
        group.add_argument('--use-shared-adapters', default=False, type=strtobool,
                           help='Shared adapters')
        group.add_argument('--use-shared-adapters-enc', default=False, type=strtobool,
                           help='Shared adapters for encoder')
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim_tgt, odim_src, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        # special tokens and model dimensions
        self.pad = 0
        self.sos_tgt = odim_tgt - 1
        self.eos_tgt = odim_tgt - 1
        self.sos_src = odim_src - 1
        self.eos_src = odim_src - 1
        self.odim_tgt = odim_tgt
        self.odim_src = odim_src
        self.idim = idim
        self.adim = args.adim
        self.ignore_id = ignore_id

        # submodule
        self.mtlalpha = getattr(args, "mtlalpha", 0.0)
        self.asr_weight = getattr(args, "asr_weight", 0.0)
        self.mt_weight = getattr(args, "mt_weight", 0.0)
        self.num_decoders = getattr(args, "num_decoders", 2)
        self.do_st = getattr(args, "do_st", True)
        self.do_mt = getattr(args, "do_mt", self.mt_weight > 0.0)
        self.do_asr = self.asr_weight > 0 and self.mtlalpha < 1

        # cross-attention parameters
        self.cross_weight = getattr(args, "cross_weight", 0.0)
        self.cross_self = getattr(args, "cross_self", False)
        self.cross_src = getattr(args, "cross_src", False)
        self.cross_operator = getattr(args, "cross_operator", None)
        self.cross_to_asr = getattr(args, "cross_to_asr", False)
        self.cross_to_st = getattr(args, "cross_to_st", False)
        self.wait_k_asr = getattr(args, "wait_k_asr", 0)
        self.wait_k_st = getattr(args, "wait_k_st", 0)
        self.cross_src_from = getattr(args, "cross_src_from", "embedding")
        self.cross_self_from = getattr(args, "cross_self_from", "embedding")
        self.cross_shared = getattr(args, "cross_shared", False)
        self.cross_weight_learnable = getattr(args, "cross_weight_learnable", False)

        # one-to-many models parameters
        self.use_joint_dict = getattr(args, "use_joint_dict", True)
        self.one_to_many = getattr(args, "one_to_many", False)
        self.use_lid = getattr(args, "use_lid", False)
        if self.use_joint_dict:
            self.langs_dict = getattr(args, "langs_dict_tgt", None)
        self.lang_tok = getattr(args, "lang_tok", None)
        self.lang_tok_mt = getattr(args, "lang_tok_mt", None)

        self.subsample = get_subsample(args, 
                                       mode='mt' if self.do_mt else 'st', 
                                       arch='transformer')
        self.reporter = MTReporter() if self.do_mt else Reporter() 
        self.normalize_before = getattr(args, "normalize_before", True)

        # Backward compatability
        if self.cross_operator in ["sum", "concat"]:
            if self.cross_self and self.cross_src:
                self.cross_operator = "self_src" + self.cross_operator
            elif self.cross_self:
                self.cross_operator = "self_" + self.cross_operator
            elif self.cross_src:
                self.cross_operator = "src_" + self.cross_operator
        if self.cross_operator:
            assert self.cross_operator in ['self_sum', 'self_concat', 'src_sum', 
                                'src_concat', 'self_src_sum', 'self_src_concat']

        # Check parameters
        if self.one_to_many:
            self.use_lid = True
        if not self.do_st:
            assert (not self.cross_to_asr) and (not self.cross_to_st)
        if self.cross_operator and 'sum' in self.cross_operator and self.cross_weight <= 0:
            assert (not self.cross_to_asr) and (not self.cross_to_st)
        if self.cross_to_asr or self.cross_to_st:
            assert self.do_st and self.do_asr
            assert self.cross_self or self.cross_src
        assert bool(self.cross_operator) == (self.do_asr and (self.cross_to_asr or self.cross_to_st))
        if self.cross_src_from != "embedding" or self.cross_self_from != "embedding":
            assert self.normalize_before
        if self.wait_k_asr > 0:
            assert self.wait_k_st == 0
        elif self.wait_k_st > 0:
            assert self.wait_k_asr == 0
        else:
            assert self.wait_k_asr == 0
            assert self.wait_k_st == 0

        logging.info("*** Cross attention parameters ***")
        if self.cross_to_asr:
            logging.info("| Cross to ASR")
        if self.cross_to_st:
            logging.info("| Cross to ST")
        if self.cross_self:
            logging.info("| Cross at Self")
        if self.cross_src:
            logging.info("| Cross at Source")
        if self.cross_to_asr or self.cross_to_st:
            logging.info(f'| Cross operator: {self.cross_operator}')
            logging.info(f'| Cross sum weight: {self.cross_weight}')
            if self.cross_src:
                logging.info(f'| Cross source from: {self.cross_src_from}')
            if self.cross_self:
                logging.info(f'| Cross self from: {self.cross_self_from}')
        logging.info(f"Use joint dictionary: {self.use_joint_dict}")
        
        if (self.cross_src_from != "embedding" and self.cross_src) \
            and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using \
                self.cross_src_from == embedding for cross at source attention.')
        if (self.cross_self_from != "embedding" and self.cross_self) \
            and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using \
                self.cross_self_from == embedding for cross at self attention.')

        # Adapters
        self.use_adapters = getattr(args, "use_adapters", False)
        self.use_adapters_in_enc = getattr(args, "use_adapters_in_enc", False)
        adapter_names = getattr(args, "adapters", None)
        adapter_reduction_factor = getattr(args, "adapter_reduction_factor", None)
        adapter_reduction_factor_enc = getattr(args, "adapter_reduction_factor_enc", adapter_reduction_factor)
        use_adapters_for_asr = getattr(args, "use_adapters_for_asr", True)
        adapter_before_src_attn = getattr(args, "adapter_before_src_attn", False)
        adapter_after_mha = getattr(args, "adapter_after_mha", False)
        use_shared_adapters = getattr(args, "use_shared_adapters", False)
        use_shared_adapters_enc = getattr(args, "use_shared_adapters_enc", False)
        # if self.use_adapters and not use_adapters_for_asr:
        #     assert not self.do_asr or \
        #         (self.do_asr and self.num_decoders != 1) or \
        #         (self.do_asr and not self.do_st) # for backward compatibility

        if adapter_names:
            if self.do_asr and not self.do_st:
                adapter_names = [str(args.char_list_src.index(f'<2{l}>')) for l in adapter_names]
            else:
                adapter_names = [str(args.char_list_tgt.index(f'<2{l}>')) for l in adapter_names]
        logging.info(f'| adapters = {adapter_names}')

        if self.do_st or self.do_asr:
            logging.info(f'Speech encoder')
            self.encoder = Encoder(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=getattr(args, "transformer_input_layer", "conv2d"),
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                adapter_names=adapter_names if self.use_adapters_in_enc else None,
                reduction_factor=adapter_reduction_factor_enc,
                adapter_after_mha=adapter_after_mha,
                shared_adapters=use_shared_adapters_enc,
            )
        if self.do_st:
            logging.info('ST decoder')
            self.decoder = Decoder(
                odim=odim_tgt,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                normalize_before=self.normalize_before,
                cross_operator=self.cross_operator if self.cross_to_st else None,
                cross_shared=self.cross_shared,
                cross_weight_learnable=self.cross_weight_learnable,
                cross_weight=self.cross_weight,
                use_output_layer=True if (self.use_joint_dict or \
                                    (self.do_st and not self.do_asr)) else False,
                adapter_names=adapter_names,
                reduction_factor=adapter_reduction_factor,
                adapter_before_src_attn=adapter_before_src_attn,
                adapter_after_mha=adapter_after_mha,
                shared_adapters=use_shared_adapters,
            )
        if self.do_asr:
            logging.info('ASR decoder')
            self.decoder_asr = Decoder(
                odim=odim_src,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                normalize_before=self.normalize_before,
                cross_operator=self.cross_operator if self.cross_to_asr else None,
                cross_shared=self.cross_shared,
                cross_weight_learnable=self.cross_weight_learnable,
                cross_weight=self.cross_weight,
                use_output_layer=True if (self.use_joint_dict or \
                                    (self.do_asr and not self.do_st)) else False,
                adapter_names=adapter_names,
                reduction_factor=adapter_reduction_factor,
                adapter_before_src_attn=adapter_before_src_attn,
                adapter_after_mha=adapter_after_mha,
                shared_adapters=use_shared_adapters,
            )
            if self.num_decoders == 1 and self.do_st:
                logging.info('*** Use shared decoders *** ')
                self.decoder_asr = self.decoder

        if not self.use_joint_dict and (self.do_st and self.do_asr):
            self.output_layer = torch.nn.Linear(args.adim, odim_tgt)
            self.output_layer_asr = torch.nn.Linear(args.adim, odim_src)

        # submodule for MT task
        if self.do_mt:
            logging.info('MT encoder')
            self.encoder_mt = Encoder(
                idim=odim_src,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                input_layer='embed',
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                padding_idx=0
            )
            if not self.do_st:
                logging.info('MT decoder')
                self.decoder_mt = Decoder(
                    odim=odim_tgt,
                    attention_dim=args.adim,
                    attention_heads=args.aheads,
                    linear_units=args.dunits,
                    num_blocks=args.dlayers,
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                    src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                    normalize_before=self.normalize_before,
                    use_output_layer=True,
                )
        self.reset_parameters(args)  # place after the submodule initialization
        if self.mtlalpha > 0.0:
            self.ctc = CTC(odim_src, args.adim, args.dropout_rate, 
                            ctc_type=args.ctc_type, reduce=True,
                            zero_infinity=True)
        else:
            self.ctc = None

        if self.asr_weight > 0 and (args.report_cer or args.report_wer):
            from espnet.nets.e2e_asr_common import ErrorCalculator
            self.error_calculator = ErrorCalculator(args.char_list_src,
                                                    args.sym_space, args.sym_blank,
                                                    args.report_cer, args.report_wer)
        elif self.do_mt and getattr(args, "report_bleu", False):
            from espnet.nets.e2e_mt_common import ErrorCalculator
            self.error_calculator = ErrorCalculator(args.char_list_tgt,
                                                    args.sym_space,
                                                    args.report_bleu)
        else:
            self.error_calculator = None
        self.rnnlm = None

        # criterion
        if self.do_st:
            self.criterion_st = LabelSmoothingLoss(self.odim_tgt, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        if self.do_asr:
            self.criterion_asr = LabelSmoothingLoss(self.odim_src, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        if self.do_mt:
            self.criterion_mt = LabelSmoothingLoss(self.odim_tgt, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
            self.normalize_length = args.transformer_length_normalized_loss  # for PPL

        # Language embedding layer
        if self.lang_tok == "encoder-pre-sum":
            self.language_embeddings = build_embedding(self.langs_dict, self.idim, 
                                                            padding_idx=self.pad)
            logging.info(f'language_embeddings: {self.language_embeddings}')

        # Backward compatability
        if self.cross_operator:
            if "sum" in self.cross_operator:
                self.cross_operator = "sum"
            if "concat" in self.cross_operator: 
                self.cross_operator = "concat"

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        logging.info(f'Initialize parameters...')
        initialize(self, args.transformer_init)
        if self.mt_weight > 0:
            logging.info(f'Initialize MT encoder and decoder...')
            torch.nn.init.normal_(self.encoder_mt.embed[0].weight, 
                                    mean=0, std=args.adim ** -0.5)
            torch.nn.init.constant_(self.encoder_mt.embed[0].weight[self.pad], 0)
            torch.nn.init.normal_(self.decoder_mt.embed[0].weight, 
                                    mean=0, std=args.adim ** -0.5)
            torch.nn.init.constant_(self.decoder_mt.embed[0].weight[self.pad], 0)

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 0. Extract target language ID
        tgt_lang_ids, tgt_lang_ids_src = None, None

        if self.do_st or self.do_mt:
            if self.use_lid: # remove target language ID in the beginning
                tgt_lang_ids = ys_pad[:, 0:1]
                ys_pad = ys_pad[:, 1:]
            ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos_tgt, self.eos_tgt, 
                                                self.ignore_id) # bs x max_lens            
            if self.lang_tok == "decoder-pre" and self.lang_tok_mt != "pre-src": 
                ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)
            ys_mask = target_mask(ys_in_pad, self.ignore_id) # bs x max_lens x max_lens

        if self.do_asr or self.do_mt:
            if self.use_lid:
                tgt_lang_ids_src = ys_pad_src[:, 0:1]
                ys_pad_src = ys_pad_src[:, 1:]
            ys_in_pad_src, ys_out_pad_src = add_sos_eos(ys_pad_src, self.sos_src, self.eos_src, 
                                                        self.ignore_id) # bs x max_lens_src  
            if self.lang_tok == "decoder-pre" and self.lang_tok_mt != "pre-tgt": # _v2 for mt_model_tgt
                ys_in_pad_src = torch.cat([tgt_lang_ids_src, ys_in_pad_src[:, 1:]], dim=1) 
            ys_mask_src = target_mask(ys_in_pad_src, self.ignore_id) # bs x max_lens_src x max_lens_src

        if self.do_mt and not self.do_st:
            ys_pad_src_mt = ys_in_pad_src[:, :max(ilens)]  # for data parallel
            ys_mask_src_mt = (~make_pad_mask(ilens.tolist())).to(ys_pad_src_mt.device).unsqueeze(-2)

        # 1. forward encoder
        if self.do_st or self.do_asr:
            xs_pad = xs_pad[:, :max(ilens)]  # for data parallel # bs x max_ilens x idim
            if self.lang_tok == "encoder-pre-sum":
                lang_embed = self.language_embeddings(tgt_lang_ids) # bs x 1 x idim
                xs_pad = xs_pad + lang_embed
            src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2) # bs x 1 x max_ilens

            enc_lang_id, enc_lang_id_src = None, None
            if self.use_adapters_in_enc:
                if self.do_asr:
                    enc_lang_id_src = str(tgt_lang_ids_src[0].data.cpu().numpy()[0])
                if self.do_st:
                    enc_lang_id = str(tgt_lang_ids[0].data.cpu().numpy()[0])                   
            # forward pass
            hs_pad, hs_mask = self.encoder(xs_pad, src_mask, enc_lang_id)
            hs_pad_src, hs_mask_src = hs_pad, hs_mask
            if self.use_adapters_in_enc and self.do_asr:
                hs_pad_src, hs_mask_src = self.encoder(xs_pad, src_mask, enc_lang_id_src)
        elif self.do_mt and not self.do_st:
            hs_pad_mt, hs_mask_mt = self.encoder_mt(ys_pad_src_mt, ys_mask_src_mt)
        else:
            raise NotImplementedError

        # 2. forward decoders
        pred_pad, pred_pad_asr, pred_pad_mt = None, None, None
        loss_att, loss_asr, loss_mt = 0.0, 0.0, 0.0

        if self.do_st:
            if self.cross_to_st:
                if self.wait_k_asr > 0:
                    cross_mask = create_cross_mask(ys_in_pad, ys_in_pad_src, 
                                        self.ignore_id, wait_k_cross=self.wait_k_asr)
                elif self.wait_k_st > 0:
                    cross_mask = create_cross_mask(ys_in_pad, ys_in_pad_src, 
                                        self.ignore_id, wait_k_cross=-self.wait_k_st)
                else:
                    cross_mask = create_cross_mask(ys_in_pad, ys_in_pad_src, 
                                        self.ignore_id, wait_k_cross=0)
                cross_input = self.decoder_asr.embed(ys_in_pad_src)
                if (self.cross_src_from == "before-self" and self.cross_src) or \
                    (self.cross_self_from == "before-self" and self.cross_self):
                    cross_input = self.decoder_asr.decoders[0].norm1(cross_input)
                pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask,
                                            cross=cross_input, cross_mask=cross_mask,
                                            cross_self=self.cross_self, cross_src=self.cross_src)
            else:
                pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

            if not self.use_joint_dict and (self.do_st and self.do_asr):
                pred_pad = self.output_layer(pred_pad)

            self.pred_pad = pred_pad
            # compute attention loss
            loss_att = self.criterion_st(pred_pad, ys_out_pad)

        # Multi-task w/ ASR  
        if self.do_asr:
            if self.cross_to_asr:
                if self.wait_k_asr > 0:
                    cross_mask = create_cross_mask(ys_in_pad_src, ys_in_pad, 
                                    self.ignore_id, wait_k_cross=-self.wait_k_asr)
                elif self.wait_k_st > 0:
                    cross_mask = create_cross_mask(ys_in_pad_src, ys_in_pad, 
                                    self.ignore_id, wait_k_cross=self.wait_k_st)
                else:
                    cross_mask = create_cross_mask(ys_in_pad_src, ys_in_pad, 
                                    self.ignore_id, wait_k_cross=0)
                cross_input = self.decoder.embed(ys_in_pad)
                if (self.cross_src_from == "before-self" and self.cross_src) or \
                    (self.cross_self_from == "before-self" and self.cross_self):
                    cross_input = self.decoder.decoders[0].norm1(cross_input)

                pred_pad_asr, _ = self.decoder_asr(ys_in_pad_src, ys_mask_src, hs_pad_src, hs_mask_src,
                                        cross=cross_input, cross_mask=cross_mask,
                                        cross_self=self.cross_self, cross_src=self.cross_src)
            else:
                pred_pad_asr, _ = self.decoder_asr(ys_in_pad_src, ys_mask_src, hs_pad_src, hs_mask_src)

            if not self.use_joint_dict and (self.do_st and self.do_asr):
                pred_pad_asr = self.output_layer_asr(pred_pad_asr)

            self.pred_pad_asr = pred_pad_asr
            # compute loss
            loss_asr = self.criterion_asr(pred_pad_asr, ys_out_pad_src)

        # Multi-task w/ MT
        if self.do_mt:
            if self.do_st:
                # forward MT encoder
                ilens_mt = torch.sum(ys_pad_src != self.ignore_id, dim=1).cpu().numpy()
                # NOTE: ys_pad_src is padded with -1
                ys_src = [y[y != self.ignore_id] for y in ys_pad_src]  # parse padded ys_src
                ys_zero_pad_src = pad_list(ys_src, self.pad)  # re-pad with zero
                ys_zero_pad_src = ys_zero_pad_src[:, :max(ilens_mt)]  # for data parallel
                src_mask_mt = (~make_pad_mask(ilens_mt.tolist())).to(ys_zero_pad_src.device).unsqueeze(-2)

                hs_pad_mt, hs_mask_mt = self.encoder_mt(ys_zero_pad_src, src_mask_mt)
                # forward MT decoder
                pred_pad_mt, _ = self.decoder(ys_in_pad, ys_mask, hs_pad_mt, hs_mask_mt)
                # compute loss
                loss_mt = self.criterion_st(pred_pad_mt, ys_out_pad)
            else:
                pred_pad_mt, pred_mask_mt = self.decoder_mt(ys_in_pad, ys_mask, hs_pad_mt, hs_mask_mt)
                loss_mt = self.criterion_mt(pred_pad_mt, ys_out_pad)

        # compute accuracy
        self.acc = th_accuracy(pred_pad.view(-1, self.odim_tgt), ys_out_pad,
                                ignore_label=self.ignore_id) if pred_pad is not None else 0.0
        self.acc_asr = th_accuracy(pred_pad_asr.view(-1, self.odim_src), ys_out_pad_src,
                                ignore_label=self.ignore_id) if pred_pad_asr is not None else 0.0
        self.acc_mt = th_accuracy(pred_pad_mt.view(-1, self.odim_tgt), ys_out_pad,
                                ignore_label=self.ignore_id) if pred_pad_mt is not None else 0.0

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0 or self.asr_weight == 0:
            loss_ctc = 0.0
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad_src)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad_src.cpu(), is_ctc=True)

        # 5. compute cer/wer
        cer, wer = None, None  # TODO(hirofumi0810): fix later
        # if self.training or (self.asr_weight == 0 or self.mtlalpha == 1 or not (self.report_cer or self.report_wer)):
        #     cer, wer = None, None
        # else:
        #     ys_hat = pred_pad.argmax(dim=-1)
        #     cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copyied from e2e_asr
        alpha = self.mtlalpha
        self.loss = (1 - self.asr_weight - self.mt_weight) * loss_att + \
            self.asr_weight * (alpha * loss_ctc + (1 - alpha) * loss_asr) + \
            self.mt_weight * loss_mt
        loss_asr_data = float(alpha * loss_ctc + (1 - alpha) * loss_asr)
        loss_mt_data = None if self.mt_weight == 0 else float(loss_mt)
        loss_st_data = float(loss_att)
        loss_data = float(self.loss)

        # compute bleu and ppl for mt model
        if self.do_mt:
            bleu = 0.0
            if self.training or self.error_calculator is None:
                bleu = 0.0
            else:
                ys_hat_mt = pred_pad_mt.argmax(dim=-1)
                bleu = self.error_calculator(ys_hat_mt.cpu(), ys_out_pad.cpu())

            if self.normalize_length:
                self.ppl = np.exp(loss_data)
            else:
                ys_out_pad = ys_out_pad.view(-1)
                ignore = ys_out_pad == self.ignore_id  # (B,)
                total = len(ys_out_pad) - ignore.sum().item()
                self.ppl = np.exp(loss_data * ys_out_pad.size(0) / total)

        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            if self.do_mt:
                self.reporter.report(loss_data, self.acc_mt, self.ppl, bleu)
            else:
                self.reporter.report(loss_asr_data, loss_mt_data, loss_st_data,
                                 self.acc_asr, self.acc_mt, self.acc,
                                 cer_ctc, cer, wer, 0.0,  # TODO(hirofumi0810): bleu
                                 loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder)

    def encode(self, x):
        """Encode source acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        if self.do_mt:
            enc_output, _ = self.encoder_mt(x, None)
        else:
            enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size_asr
        penalty = recog_args.penalty_asr
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos_src
        if self.use_lid and self.lang_tok == 'decoder-pre':
            src_lang_id = '<2{}>'.format(recog_args.config.split('.')[-2].split('-')[0])
            y = char_list.index(src_lang_id)
            logging.info(f'src_lang_id: {src_lang_id} - y: {y}')

        logging.info(f'y: {y}')
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio_asr == 0:
            maxlen = h.shape[0]
        else:
            maxlen = max(1, int(recog_args.maxlenratio_asr * h.size(0)))
        minlen = int(recog_args.minlenratio_asr * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:
            import numpy

            from espnet.nets.ctc_prefix_score import CTCPrefixScore

            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos_src, numpy)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []
       
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder_asr.forward_one_step(ys, ys_mask, enc_output)[0]
                if not self.use_joint_dict and (self.do_st and self.do_asr):
                    local_att_scores = self.output_layer_asr(local_att_scores)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos_src)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos_src:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection            
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            logging.debug(f'hyps remained: {hyps}')
            if len(hyps) > 0:
                logging.info('remained hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.info(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.info('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def translate(self, x, trans_args, char_list=None, rnnlm=None, use_jit=False):
        """Translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos_tgt

        if self.use_lid and self.lang_tok == 'decoder-pre':
            if self.lang_tok_mt is None or self.lang_tok_mt == "pre-tgt":
                tgt_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[-1])
                src_lang_id = self.sos_src
                y = char_list.index(tgt_lang_id)
            elif self.lang_tok_mt == "pre-src":
                tgt_lang_id = self.sos_tgt
                src_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[-1])

            if self.do_mt:
                if src_lang_id != self.sos_src:
                    src_lang_id = char_list.index(src_lang_id)
                x[0].insert(0, src_lang_id)
            
            logging.info(f'tgt_lang_id: {tgt_lang_id} - y: {y}')
            logging.info(f'src_lang_id: {src_lang_id}')

        logging.info('<sos> index: ' + str(y))
        logging.info('<sos> mark: ' + char_list[y])

        if self.do_mt:
            x = to_device(self, torch.from_numpy(np.fromiter(map(int, x[0]), dtype=np.int64)))
            xs_pad = x.unsqueeze(0)

        enc_output = self.encode(x).unsqueeze(0)
        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        vy = h.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * h.size(0)))
        minlen = int(trans_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        hyps = [hyp]
        ended_hyps = []
     
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    if self.do_mt:
                        local_att_scores = self.decoder_mt.forward_one_step(ys, ys_mask, enc_output)[0]
                    else:
                        local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output)[0]
                if not self.use_joint_dict and (self.do_st and self.do_asr):
                    local_att_scores = self.output_layer(local_att_scores)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + trans_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.info(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos_tgt)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos_tgt:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += trans_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection            
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), trans_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.translate(x, trans_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def update_hypothesis(self, i, beam, hyps, hyps_cross_candidates,
                          enc_output, decoder, eos_cross, decoder_cross=None, wait_k_cross=0):
        hyps_best_kept = []
        # For each ST hypothesis, we use the best ASR candidates as cross information
        for hyp in hyps:
            # get nbest local scores and their ids
            ys_mask = subsequent_mask(i + 1).unsqueeze(0)
            ys = torch.tensor(hyp['yseq']).unsqueeze(0)
            # FIXME: jit does not match non-jit result
            
            if decoder_cross is not None:
                all_scores = []
                for hyp_cross in hyps_cross_candidates:
                    if len(hyp_cross) > 2:
                        if hyp_cross[-1] == eos_cross and hyp_cross[-2] == eos_cross:
                            hyp_cross.append(eos_cross)
                    ys_cross = torch.tensor(hyp_cross['yseq']).unsqueeze(0)
                    y_cross = decoder_cross.embed(ys_cross)
                    if (self.cross_self_from == "before-self" and self.cross_self) or \
                        (self.cross_src_from == "before-self" and self.cross_src):
                        y_cross = decoder_cross.decoders[0].norm1(y_cross)
                    cross_mask = create_cross_mask(ys, ys_cross, self.ignore_id, wait_k_cross=wait_k_cross)
                    local_att_scores = decoder.forward_one_step(ys, ys_mask, enc_output, 
                                                                cross=y_cross, cross_mask=cross_mask,
                                                                cross_self=self.cross_self, cross_src=self.cross_src)[0]
                    V = local_att_scores.shape[-1]
                    all_scores.append(local_att_scores)
                local_scores = torch.cat(all_scores, dim=-1)
                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)
                local_best_ids = local_best_ids % V
            else:
                local_scores = decoder.forward_one_step(ys, ys_mask, enc_output)[0]
                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

            for j in six.moves.range(beam):
                new_hyp = {}
                new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                # will be (2 x beam) hyps at most
                hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(
                hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

        return hyps_best_kept

    def process_hypothesis(self, i, hyps, ended_hyps, maxlen, minlen, trans_args, eos, rnnlm=None):

        stop_decoding = False

        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info('adding <eos> in the last postion in the loop')
            for hyp in hyps:
                hyp['yseq'].append(eos)

        # add ended hypothes to a final list, and removed them from current hypothes
        # (this will be a probmlem, number of hyps < beam)
        remained_hyps = []
        for hyp in hyps:
            if hyp['yseq'][-1] == eos:
                # only store the sequence that has more than minlen outputs
                # also add penalty
                if len(hyp['yseq']) > minlen:
                    hyp['score'] += (i + 1) * trans_args.penalty
                    if rnnlm:  # Word LM needs to add final <eos> score
                        hyp['score'] += trans_args.lm_weight * rnnlm.final(
                            hyp['rnnlm_prev'])
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)

        # end detection      
        if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
            logging.info('end detected at %d', i)
            stop_decoding = True

        hyps = remained_hyps
        if len(hyps) > 0:
            logging.debug('remained hypothes: ' + str(len(hyps)))
        else:
            logging.info('no hypothesis. Finish decoding.')
            stop_decoding = True
        
        return hyps, ended_hyps, stop_decoding

    def recognize_and_translate_separate(self, x, trans_args, 
                                        char_list_tgt=None,
                                        char_list_src=None,
                                        rnnlm=None,
                                        use_jit=False):
        """Translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list_tgt: list of characters for target languages
        :param list char_list_src: list of characters for source languages
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list_tgt.index(trans_args.tgt_lang)
        else:
            y = self.sos_tgt

        if self.one_to_many and self.lang_tok == 'decoder-pre':
            tgt_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[-1])
            y = char_list_tgt.index(tgt_lang_id)
            logging.info(f'tgt_lang_id: {tgt_lang_id} - y: {y}')

            src_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[0])
            y_asr = char_list_src.index(src_lang_id)
            logging.info(f'src_lang_id: {src_lang_id} - y_asr: {y_asr}')
        else:
            y = self.sos_tgt
            y_asr = self.sos_src
        logging.info(f'<sos> index: {str(y)}; <sos> mark: {char_list_tgt[y]}')
        logging.info(f'<sos> index asr: {str(y_asr)}; <sos> mark asr: {char_list_src[y_asr]}')

        enc_output = self.encode(x).unsqueeze(0)
        h = enc_output.squeeze(0)
        logging.info('input lengths: ' + str(h.size(0)))

        # search parms
        beam = trans_args.beam_size
        beam_cross = trans_args.beam_cross_size
        penalty = trans_args.penalty
        assert beam_cross <= beam

        if trans_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            maxlen = max(1, int(trans_args.maxlenratio * h.size(0)))
        if trans_args.maxlenratio_asr == 0:
            maxlen_asr = h.shape[0]
        else:
            maxlen_asr = max(1, int(trans_args.maxlenratio_asr * h.size(0)))
        minlen = int(trans_args.minlenratio * h.size(0))
        minlen_asr = int(trans_args.minlenratio_asr * h.size(0))
        logging.info(f'max output length: {str(maxlen)}; min output length: {str(minlen)}')
        logging.info(f'max output length asr: {str(maxlen_asr)}; min output length asr: {str(minlen_asr)}')

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
            hyp_asr = {'score': 0.0, 'yseq': [y_asr]}

        hyps = [hyp]
        hyps_asr = [hyp_asr]
        ended_hyps = []
        ended_hyps_asr = []
        stop_decoding_st = False
        stop_decoding_asr = False
        hyps_st_candidates = hyps
        hyps_asr_candidates = hyps_asr
        
        traced_decoder = None
        for i in six.moves.range(maxlen + self.wait_k_asr):
            logging.info('position ' + str(i))

            # Start ASR first, then after self.wait_k_asr steps, start ST
            # ASR SEARCH
            if i < maxlen and not stop_decoding_asr:
                decoder_cross = self.decoder if self.cross_to_asr else None
                hyps_asr = self.update_hypothesis(i, beam, 
                                                  hyps_asr, 
                                                  hyps_st_candidates,
                                                  enc_output, 
                                                  self.eos_tgt, 
                                                  decoder=self.decoder_asr, 
                                                  decoder_cross=decoder_cross, 
                                                  wait_k_cross=self.wait_k_st
                                                  )
                hyps_asr, ended_hyps_asr, stop_decoding_asr = self.process_hypothesis(i, 
                                                                    hyps_asr,
                                                                    ended_hyps_asr,
                                                                    maxlen_asr,
                                                                    minlen_asr,
                                                                    trans_args,
                                                                    self.eos_src,
                                                                    rnnlm=rnnlm
                                                                    )
                hyps_asr_candidates = sorted(hyps_asr + ended_hyps_asr,
                                            key=lambda x: x['score'], reverse=True)[:beam_cross]
                if char_list_src is not None:
                    for hyp in hyps_asr:
                        logging.info('hypo asr: ' + ''.join([char_list_src[int(x)] for x in hyp['yseq']]))
            # ST SEARCH
            if i >= self.wait_k_asr and not stop_decoding_st:
                decoder_cross = self.decoder_asr if self.cross_to_st else None
                hyps = self.update_hypothesis(i - self.wait_k_asr, beam, 
                                              hyps, hyps_asr_candidates,
                                              enc_output,
                                              self.eos_src,
                                              decoder=self.decoder,
                                              decoder_cross=decoder_cross,
                                              wait_k_cross=self.wait_k_asr
                                              )
                hyps, ended_hyps, stop_decoding_st = self.process_hypothesis(i - self.wait_k_asr, 
                                                            hyps, ended_hyps,
                                                            maxlen, minlen,
                                                            trans_args,
                                                            self.eos_tgt,
                                                            rnnlm=rnnlm
                                                            )
                hyps_st_candidates = sorted(hyps + ended_hyps,
                                            key=lambda x: x['score'], reverse=True)[:beam_cross]
                if char_list_tgt is not None:
                    for hyp in hyps:
                        logging.info('hypo: ' + ''.join([char_list_tgt[int(x)] for x in hyp['yseq']]))
            
            # Stop decoding check
            if stop_decoding_asr and stop_decoding_st:
                break
        
        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), trans_args.nbest)]

        logging.info('best hypo: ' + ''.join([char_list_tgt[int(x)] for x in nbest_hyps[0]['yseq']]))

        nbest_hyps_asr = sorted(
            ended_hyps_asr, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps_asr), trans_args.nbest)]

        logging.info('best hypo asr: ' + ''.join([char_list_src[int(x)] for x in nbest_hyps_asr[0]['yseq']]))

        # check number of hypotheis
        if len(nbest_hyps) == 0 or len(nbest_hyps_asr) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.recognize_and_translate(x, trans_args, char_list_tgt, char_list_src, rnnlm, use_jit)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        
        return nbest_hyps, nbest_hyps_asr

    def recognize_and_translate_sum(self, x, trans_args, 
                                    char_list_tgt=None,
                                    char_list_src=None, 
                                    rnnlm=None, 
                                    use_jit=False, 
                                    decode_asr_weight=1.0, 
                                    score_is_prob=False, 
                                    ratio_diverse_st=0.0,
                                    ratio_diverse_asr=0.0,
                                    use_rev_triu_width=0,
                                    use_diag=False,
                                    debug=False):
        """Recognize and translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        assert self.do_asr, "Recognize and translate are performed simultaneously."

        self.wait_k_asr = max(self.wait_k_asr, getattr(trans_args, "wait_k_asr", 0))
        self.wait_k_st = max(self.wait_k_st, getattr(trans_args, "wait_k_st", 0)) 

        logging.info(f'| ratio_diverse_st = {ratio_diverse_st}')
        logging.info(f'| ratio_diverse_asr = {ratio_diverse_asr}')
        logging.info(f'| wait_k_asr = {self.wait_k_asr}')
        logging.info(f'| wait_k_st = {self.wait_k_st}')

        # prepare sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list_tgt.index(trans_args.tgt_lang)
        else:
            y = self.sos_tgt

        if self.use_lid and self.lang_tok == 'decoder-pre':
            tgt_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[-1])
            y = char_list_tgt.index(tgt_lang_id)
            logging.info(f'tgt_lang_id: {tgt_lang_id} - y: {y}')

            src_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[0])
            y_asr = char_list_src.index(src_lang_id)
            logging.info(f'src_lang_id: {src_lang_id} - y_asr: {y_asr}')
        else:
            y = self.sos_tgt
            y_asr = self.sos_src
        
        logging.info(f'<sos> index: {str(y)}; <sos> mark: {char_list_tgt[y]}')
        logging.info(f'<sos> index asr: {str(y_asr)}; <sos> mark asr: {char_list_src[y_asr]}')

        enc_output = self.encode(x).unsqueeze(0)
        h = enc_output.squeeze(0)
        logging.info('input lengths: ' + str(h.size(0)))

        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        vy = h.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            maxlen = max(1, int(trans_args.maxlenratio * h.size(0)))
        if trans_args.maxlenratio_asr == 0:
            maxlen_asr = h.shape[0]
        else:
            maxlen_asr = max(1, int(trans_args.maxlenratio_asr * h.size(0)))
        minlen = int(trans_args.minlenratio * h.size(0))
        minlen_asr = int(trans_args.minlenratio_asr * h.size(0))
        logging.info(f'max output length: {str(maxlen)}; min output length: {str(minlen)}')
        logging.info(f'max output length asr: {str(maxlen_asr)}; min output length asr: {str(minlen_asr)}')

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            logging.info('initializing hypothesis...')
            hyp = {'score': 0.0, 'yseq': [y], 'yseq_asr': [y_asr]}

        hyps = [hyp]
        ended_hyps = []

        traced_decoder = None
        for i in six.moves.range(max(maxlen, maxlen_asr)):
            logging.info('position ' + str(i))

            hyps_best_kept = []

            for idx, hyp in enumerate(hyps):
                
                if self.wait_k_asr > 0:
                    if i < self.wait_k_asr:
                        ys_mask = subsequent_mask(1).unsqueeze(0)
                    else:
                        ys_mask = subsequent_mask(i - self.wait_k_asr + 1).unsqueeze(0)
                else:
                    ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)

                if self.wait_k_st > 0:
                    if i < self.wait_k_st:
                        ys_mask_asr = subsequent_mask(1).unsqueeze(0)
                    else:
                        ys_mask_asr = subsequent_mask(i - self.wait_k_st + 1).unsqueeze(0)
                else:
                    ys_mask_asr = subsequent_mask(i + 1).unsqueeze(0)
                ys_asr = torch.tensor(hyp['yseq_asr']).unsqueeze(0)

                start = time.time()
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    if (hyp['yseq_asr'][-1] != self.eos_src or i < 2) and i >= self.wait_k_st:
                        if self.cross_to_asr:
                            y_cross = self.decoder.embed(ys)
                            cross_mask_asr = create_cross_mask(ys_asr, ys, self.ignore_id, wait_k_cross=self.wait_k_st)
                            if (self.cross_self_from == "before-self" and self.cross_self) or \
                                (self.cross_src_from == "before-self" and self.cross_src):
                                 y_cross = self.decoder.decoders[0].norm1(y_cross)
                            local_att_scores_asr = self.decoder_asr.forward_one_step(ys_asr, ys_mask_asr, enc_output,
                                                                        cross=y_cross, cross_mask=cross_mask_asr,
                                                                        cross_self=self.cross_self, cross_src=self.cross_src)[0]
                        else:
                            local_att_scores_asr = self.decoder_asr.forward_one_step(ys_asr, ys_mask_asr, enc_output)[0]

                        # If using 2 separate dictionaries
                        if not self.use_joint_dict and (self.do_st and self.do_asr):
                            local_att_scores_asr = self.output_layer_asr(local_att_scores_asr)
                        if score_is_prob:
                            local_att_scores_asr = torch.exp(local_att_scores_asr)
                    else:
                        local_att_scores_asr = None

                    if (hyp['yseq'][-1] != self.eos_tgt or i < 2) and i >= self.wait_k_asr:
                        if self.cross_to_st:
                            cross_mask = create_cross_mask(ys, ys_asr, self.ignore_id, wait_k_cross=self.wait_k_asr)
                            y_cross = self.decoder_asr.embed(ys_asr)
                            if (self.cross_self_from == "before-self" and self.cross_self) or \
                                (self.cross_src_from == "before-self" and self.cross_src):
                                y_cross = self.decoder_asr.decoders[0].norm1(y_cross)
                            local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output, 
                                                                        cross=y_cross, cross_mask=cross_mask,
                                                                        cross_self=self.cross_self, cross_src=self.cross_src)[0]
                        else:
                            local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output)[0]

                        # If using 2 separate dictionaries
                        if not self.use_joint_dict and (self.do_st and self.do_asr):
                            local_att_scores = self.output_layer(local_att_scores)
                        if score_is_prob:
                            local_att_scores = torch.exp(local_att_scores)
                    else:
                        local_att_scores = None

                start = time.time()
                if local_att_scores is not None and local_att_scores_asr is not None:
                    local_att_scores_asr = decode_asr_weight * local_att_scores_asr
                    xk, ixk = local_att_scores.topk(beam)
                    yk, iyk = local_att_scores_asr.topk(beam)
                    S = (torch.mm(torch.t(xk), torch.ones_like(xk))
                                    + torch.mm(torch.t(torch.ones_like(yk)), yk))
                    s2v = torch.LongTensor([[i, j] for i in ixk.squeeze(0) for j in iyk.squeeze(0)]) # (k^2) x 2

                    # Do not force diversity
                    if ratio_diverse_st <= 0 and ratio_diverse_asr <=0:
                        local_best_scores, id2k = S.view(-1).topk(beam)
                        I = s2v[id2k]
                        local_best_ids_st = I[:,0]
                        local_best_ids_asr = I[:,1]

                    # Force diversity for ST only
                    if ratio_diverse_st > 0 and ratio_diverse_asr <= 0:
                        ct = int((1 - ratio_diverse_st) * beam)
                        s2v = s2v.reshape(beam, beam, 2)
                        Sc = S[:, :ct]
                        local_best_scores, id2k = Sc.flatten().topk(beam)
                        I = s2v[:, :ct]
                        I = I.reshape(-1, 2)
                        I = I[id2k]
                        local_best_ids_st = I[:,0]
                        local_best_ids_asr = I[:,1]

                    # Force diversity for ASR only
                    if ratio_diverse_asr > 0 and ratio_diverse_st <= 0:
                        cr = int((1 - ratio_diverse_asr) * beam)
                        s2v = s2v.reshape(beam, beam, 2)
                        Sc = S[:cr, :]
                        local_best_scores, id2k = Sc.view(-1).topk(beam)
                        I = s2v[:cr, :]
                        I = I.reshape(-1, 2)
                        I = I[id2k]
                        local_best_ids_st = I[:,0]
                        local_best_ids_asr = I[:,1]

                    # Force diversity for both ST and ASR
                    if ratio_diverse_st > 0 and ratio_diverse_asr > 0:
                        cr = int((1 - ratio_diverse_asr) * beam) 
                        ct = int((1 - ratio_diverse_st) * beam)
                        ct = max(ct, math.ceil(beam // cr))
                                    
                        s2v = s2v.reshape(beam, beam, 2)
                        Sc = S[:cr, :ct]
                        local_best_scores, id2k = Sc.flatten().topk(beam)
                        I = s2v[:cr, :ct]
                        I = I.reshape(-1, 2)
                        I = I[id2k]
                        local_best_ids_st = I[:,0]
                        local_best_ids_asr = I[:,1]

                elif local_att_scores is not None:
                    local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                    local_best_scores = local_best_scores.squeeze(0)
                    local_best_ids_st = local_best_ids_st.squeeze(0)
                elif local_att_scores_asr is not None:
                    local_best_scores, local_best_ids_asr = torch.topk(local_att_scores_asr, beam, dim=1)
                    local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    local_best_scores = local_best_scores.squeeze(0) 
                else:
                    raise NotImplementedError
                
                if debug:
                    logging.info(f'score time = {time.time() - start}')

                start = time.time()
                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[j])

                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']

                    new_hyp['yseq_asr'] = [0] * (1 + len(hyp['yseq_asr']))
                    new_hyp['yseq_asr'][:len(hyp['yseq_asr'])] = hyp['yseq_asr']

                    if local_att_scores is not None:
                        new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids_st[j])
                    else:
                        if i >= self.wait_k_asr:
                            new_hyp['yseq'][len(hyp['yseq'])] = self.eos_tgt
                        else:
                            new_hyp['yseq'] = hyp['yseq']
                    
                    if local_att_scores_asr is not None:
                        new_hyp['yseq_asr'][len(hyp['yseq_asr'])] = int(local_best_ids_asr[j])
                    else:
                        if i >= self.wait_k_st:
                            new_hyp['yseq_asr'][len(hyp['yseq_asr'])] = self.eos_src
                        else:
                            new_hyp['yseq_asr'] = hyp['yseq_asr']

                    hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]  

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    if hyp['yseq'][-1] != self.eos_tgt:
                        hyp['yseq'].append(self.eos_tgt)
            if i == maxlen_asr - 1:
                logging.info('adding <eos> in the last postion in the loop for asr')
                for hyp in hyps:
                    if hyp['yseq_asr'][-1] != self.eos_src:
                        hyp['yseq_asr'].append(self.eos_src)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []

            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos_tgt and hyp['yseq_asr'][-1] == self.eos_src:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen and len(hyp['yseq_asr']) > minlen_asr:
                        hyp['score'] += (i + 1) * penalty
                        # if rnnlm:  # Word LM needs to add final <eos> score
                        #     hyp['score'] += trans_args.lm_weight * rnnlm.final(
                        #         hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection          
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.info('remained hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list_tgt is not None and char_list_src is not None:
                for hyp in hyps:
                    logging.info('hypo: ' + ''.join([char_list_tgt[int(x)] for x in hyp['yseq']]))
                    logging.info('hypo asr: ' + ''.join([char_list_src[int(x)] for x in hyp['yseq_asr']]))
                logging.info('best hypo: ' + ''.join([char_list_tgt[int(x)] for x in hyps[0]['yseq']]))
                logging.info('best hypo asr: ' + ''.join([char_list_src[int(x)] for x in hyps[0]['yseq_asr']]))
            logging.info('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), trans_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            trans_args.minlenratio_asr = max(0.0, trans_args.minlenratio_asr - 0.1)
            return self.recognize_and_translate_sum(x, trans_args, char_list_tgt, char_list_src, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention) and m.attn is not None:  # skip MHA for submodules
                ret[name] = m.attn.cpu().numpy()
        return ret