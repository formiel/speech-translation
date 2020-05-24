# Copyright 2019 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool

import logging
import math
import six

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_st import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_dual import DualDecoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask, create_cross_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.st_interface import STInterface
from espnet.nets.e2e_asr_common import end_detect


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
    

def build_embedding(dictionary, embed_dim, padding_idx=0):
    # num_embeddings = len(dictionary)
    num_embeddings = max(list(dictionary.values())) + 1
    emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
    return emb


class E2EDualDecoder(STInterface, torch.nn.Module):
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
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
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
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )

        # submodule for ASR task
        self.mtlalpha = args.mtlalpha
        self.asr_weight = getattr(args, "asr_weight", 0.0)
        self.do_asr = self.asr_weight > 0 and args.mtlalpha < 1

        # cross-attention parameters
        self.cross_weight = getattr(args, "cross_weight", 0.0)
        self.cross_self = getattr(args, "cross_self", False)
        self.cross_src = getattr(args, "cross_src", False)
        self.cross_operator = getattr(args, "cross_operator", None)
        self.cross_to_asr = getattr(args, "cross_to_asr", False)
        self.cross_to_st = getattr(args, "cross_to_st", False)
        self.num_decoders = getattr(args, "num_decoders", 1)
        self.wait_k_asr = getattr(args, "wait_k_asr", 0)
        self.cross_src_from = getattr(args, "cross_src_from", "embedding")
        self.cross_self_from = getattr(args, "cross_self_from", "embedding")

        # one-to-many ST experiments
        self.one_to_many = getattr(args, "one_to_many", False)
        self.langs_dict = getattr(args, "langs_dict", None)
        self.lang_tok = getattr(args, "lang_tok", None)

        self.normalize_before = getattr(args, "normalize_before", True)
        logging.info(f'self.normalize_before = {self.normalize_before}')

        # Check parameters
        if self.cross_operator == 'sum' and self.cross_weight <= 0:
            assert (not self.cross_to_asr) and (not self.cross_to_st)
        if self.cross_to_asr or self.cross_to_st:
            assert self.do_asr
            assert self.cross_self or self.cross_src
        assert bool(self.cross_operator) == (self.do_asr and (self.cross_to_asr or self.cross_to_st))
        if self.cross_src_from != "embedding" or self.cross_self_from != "embedding":
            assert self.normalize_before

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
        
        if (self.cross_src_from != "embedding" and self.cross_src) and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using self.cross_src_from == embedding for cross at source attention.')
        if (self.cross_self_from != "embedding" and self.cross_self) and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using self.cross_self_from == embedding for cross at self attention.')

        # self.decoder = Decoder(
        #     odim=odim,
        #     attention_dim=args.adim,
        #     attention_heads=args.aheads,
        #     linear_units=args.dunits,
        #     num_blocks=args.dlayers,
        #     dropout_rate=args.dropout_rate,
        #     positional_dropout_rate=args.dropout_rate,
        #     self_attention_dropout_rate=args.transformer_attn_dropout_rate,
        #     src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        #     normalize_before=self.normalize_before,
        #     cross_operator=self.cross_operator
        # )
        self.dual_decoder = DualDecoder(
                odim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                normalize_before=self.normalize_before,
                cross_operator=self.cross_operator
        )

        self.pad = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.idim = idim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode='st', arch='transformer')
        self.reporter = Reporter()

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        # self.verbose = args.verbose
        self.adim = args.adim

        # if self.do_asr:
        #     logging.info('*** Do ASR ***')
        #     self.decoder_asr = Decoder(
        #         odim=odim,
        #         attention_dim=args.adim,
        #         attention_heads=args.aheads,
        #         linear_units=args.dunits,
        #         num_blocks=args.dlayers,
        #         dropout_rate=args.dropout_rate,
        #         positional_dropout_rate=args.dropout_rate,
        #         self_attention_dropout_rate=args.transformer_attn_dropout_rate,
        #         src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        #         normalize_before=self.normalize_before,
        #         cross_operator=self.cross_operator
        #     )
        #     if self.num_decoders == 1:
        #         logging.info('*** Use 1 decoder *** ')
        #         self.decoder_asr = self.decoder
        
        # submodule for MT task
        self.mt_weight = getattr(args, "mt_weight", 0.0)
        if self.mt_weight > 0:
            self.encoder_mt = Encoder(
                idim=odim,
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
        self.reset_parameters(args)  # place after the submodule initialization
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        if self.asr_weight > 0 and (args.report_cer or args.report_wer):
            from espnet.nets.e2e_asr_common import ErrorCalculator
            self.error_calculator = ErrorCalculator(args.char_list,
                                                    args.sym_space, args.sym_blank,
                                                    args.report_cer, args.report_wer)
        else:
            self.error_calculator = None
        self.rnnlm = None

        # multilingual E2E-ST related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = getattr(args, "replace_sos", False)
        if self.multilingual:
            assert self.replace_sos

        if self.lang_tok == "encoder-pre-sum":
            self.language_embeddings = build_embedding(self.langs_dict, self.idim, padding_idx=self.pad)
            print(f'language_embeddings: {self.language_embeddings}')

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)
        if self.mt_weight > 0:
            torch.nn.init.normal_(self.encoder_mt.embed[0].weight, mean=0, std=args.adim ** -0.5)
            torch.nn.init.constant_(self.encoder_mt.embed[0].weight[self.pad], 0)

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
        # src_lang_ids = None
        tgt_lang_ids, tgt_lang_ids_src = None, None
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining

        if self.one_to_many:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
            
            if self.do_asr:
                tgt_lang_ids_src = ys_pad_src[:, 0:1]
                ys_pad_src = ys_pad_src[:, 1:]  # remove target language ID in the beggining

        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel # bs x max_ilens x idim

        if self.lang_tok == "encoder-pre-sum":
            lang_embed = self.language_embeddings(tgt_lang_ids) # bs x 1 x idim
            xs_pad = xs_pad + lang_embed

        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2) # bs x 1 x max_ilens
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask) # hs_pad: bs x (max_ilens/4) x adim; hs_mask: bs x 1 x (max_ilens/4)
        self.hs_pad = hs_pad

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id) # bs x max_lens

        if self.do_asr:
            ys_in_pad_src, ys_out_pad_src = add_sos_eos(ys_pad_src, self.sos, self.eos, self.ignore_id) # bs x max_lens_src

        # replace <sos> with target language ID
        if self.replace_sos:
            ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)

        if self.lang_tok == "decoder-pre":
            ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)
            if self.do_asr:
                ys_in_pad_src = torch.cat([tgt_lang_ids_src, ys_in_pad_src[:, 1:]], dim=1)

        ys_mask = target_mask(ys_in_pad, self.ignore_id) # bs x max_lens x max_lens
        # logging.info(f'ys_in_pad.size() = {ys_in_pad.size()}')
        # logging.info(f'ys_in_pad_src.size() = {ys_in_pad_src.size()}')
        # logging.info(f'ys_mask.size() = {ys_mask.size()}')

        if self.do_asr:
            ys_mask_src = target_mask(ys_in_pad_src, self.ignore_id) # bs x max_lens x max_lens_src
            # logging.info(f'ys_mask_src.size() = {ys_mask_src.size()}')

        # if self.cross_to_st:
        #     cross_mask = create_cross_mask(ys_in_pad, ys_in_pad_src, self.ignore_id, wait_k_cross=self.wait_k_asr)
        #     # logging.info(f'st cross_mask \t {cross_mask}')
        #     # logging.info(f'st cross_mask.size() = {cross_mask.size()}')
        #     cross_input = self.decoder_asr.embed(ys_in_pad_src)
        #     if (self.cross_src_from == "before-self" and self.cross_src) or \
        #             (self.cross_self_from == "before-self" and self.cross_self): 
        #         cross_input = self.decoder_asr.decoders[0].norm1(cross_input)
        #     # logging.info(f'st cross_input size = {cross_input.size()}')
        #     # logging.info(f'st cross_input norm = {torch.norm(cross_input)}')
        #     pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask,
        #                                 cross=cross_input, cross_mask=cross_mask,
        #                                 cross_self=self.cross_self, cross_src=self.cross_src,
        #                                 cross_operator=self.cross_operator, cross_weight=self.cross_weight)
        # else:
        #     pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        cross_mask = create_cross_mask(ys_in_pad, ys_in_pad_src, self.ignore_id, wait_k_cross=self.wait_k_asr)
        cross_mask_asr = create_cross_mask(ys_in_pad_src, ys_in_pad, self.ignore_id, wait_k_cross=0)
        pred_pad, pred_mask, pred_pad_asr, pred_mask_asr = self.dual_decoder(ys_in_pad, ys_mask, ys_in_pad_src, ys_mask_src,
                                                                                hs_pad, hs_mask, cross_mask, cross_mask_asr,
                                                                                cross_self=self.cross_self, cross_src=self.cross_src,
                                                                                cross_self_from=self.cross_self_from,
                                                                                cross_src_from=self.cross_src_from,
                                                                                cross_operator=self.cross_operator,
                                                                                cross_weight=self.cross_weight)

        self.pred_pad = pred_pad
        self.pred_pad_asr = pred_pad_asr
        pred_pad_mt = None

        # 3. compute attention loss
        loss_asr, loss_mt = 0.0, 0.0
        loss_att = self.criterion(pred_pad, ys_out_pad)
        # # Multi-task w/ ASR
        # if self.do_asr:
        #     # forward ASR decoder
        #     if self.cross_to_asr:
        #         cross_mask = create_cross_mask(ys_in_pad_src, ys_in_pad, self.ignore_id, wait_k_cross=0)
        #         # logging.info(f'asr cross_mask \t {cross_mask}')
        #         # logging.info(f'asr cross_mask.size() = {cross_mask.size()}')
        #         cross_input = self.decoder.embed(ys_in_pad)
        #         if (self.cross_src_from == "before-self" and self.cross_src) or \
        #             (self.cross_self_from == "before-self" and self.cross_self):
        #             cross_input = self.decoder.decoders[0].norm1(cross_input)
        #         # logging.info(f'asr cross_input size = {cross_input.size()}')
        #         # logging.info(f'asr cross_input norm = {torch.norm(cross_input)}')

        #         pred_pad_asr, _ = self.decoder_asr(ys_in_pad_src, ys_mask_src, hs_pad, hs_mask,
        #                                 cross=cross_input, cross_mask=cross_mask,
        #                                 cross_self=self.cross_self, cross_src=self.cross_src,
        #                                 cross_operator=self.cross_operator, cross_weight=self.cross_weight)
        #     else:
        #         pred_pad_asr, _ = self.decoder_asr(ys_in_pad_src, ys_mask_src, hs_pad, hs_mask)
        # compute loss
        loss_asr = self.criterion(pred_pad_asr, ys_out_pad_src)
        # Multi-task w/ MT
        if self.mt_weight > 0:
            # forward MT encoder
            ilens_mt = torch.sum(ys_pad_src != self.ignore_id, dim=1).cpu().numpy()
            # NOTE: ys_pad_src is padded with -1
            ys_src = [y[y != self.ignore_id] for y in ys_pad_src]  # parse padded ys_src
            ys_zero_pad_src = pad_list(ys_src, self.pad)  # re-pad with zero
            ys_zero_pad_src = ys_zero_pad_src[:, :max(ilens_mt)]  # for data parallel
            src_mask_mt = (~make_pad_mask(ilens_mt.tolist())).to(ys_zero_pad_src.device).unsqueeze(-2)
            # ys_zero_pad_src, ys_pad = self.target_forcing(ys_zero_pad_src, ys_pad)
            hs_pad_mt, hs_mask_mt = self.encoder_mt(ys_zero_pad_src, src_mask_mt)
            # forward MT decoder
            pred_pad_mt, _ = self.decoder(ys_in_pad, ys_mask, hs_pad_mt, hs_mask_mt)
            # compute loss
            loss_mt = self.criterion(pred_pad_mt, ys_out_pad)

        self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                               ignore_label=self.ignore_id)
        if pred_pad_asr is not None:
            self.acc_asr = th_accuracy(pred_pad_asr.view(-1, self.odim), ys_out_pad_src,
                                       ignore_label=self.ignore_id)
        else:
            self.acc_asr = 0.0
        if pred_pad_mt is not None:
            self.acc_mt = th_accuracy(pred_pad_mt.view(-1, self.odim), ys_out_pad,
                                      ignore_label=self.ignore_id)
        else:
            self.acc_mt = 0.0

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
        self.loss = (1 - self.asr_weight - self.mt_weight) * loss_att + self.asr_weight * \
            (alpha * loss_ctc + (1 - alpha) * loss_asr) + self.mt_weight * loss_mt
        loss_asr_data = float(alpha * loss_ctc + (1 - alpha) * loss_asr)
        loss_mt_data = None if self.mt_weight == 0 else float(loss_mt)
        loss_st_data = float(loss_att)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
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
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize_and_translate_sum(self, x, trans_args, 
                                    char_list=None, rnnlm=None, use_jit=False, 
                                    decode_asr_weight=1.0, 
                                    score_is_prob=False, 
                                    ratio_diverse_st=0.0,
                                    ratio_diverse_asr=0.0,
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
        logging.info(f'| ratio_diverse_st = {ratio_diverse_st}')
        logging.info(f'| ratio_diverse_asr = {ratio_diverse_asr}')

        # prepare sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos

        if self.one_to_many and self.lang_tok == 'decoder-pre':
            tgt_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[-1])
            y = char_list.index(tgt_lang_id)
            logging.info(f'tgt_lang_id: {tgt_lang_id} - y: {y}')

            src_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[0])
            y_asr = char_list.index(src_lang_id)
            logging.info(f'src_lang_id: {src_lang_id} - y_asr: {y_asr}')
        else:
            y = self.sos
            y_asr = self.sos
        
        logging.info(f'<sos> index: {str(y)}; <sos> mark: {char_list[y]}')
        logging.info(f'<sos> index asr: {str(y_asr)}; <sos> mark asr: {char_list[y_asr]}')

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
                # get nbest local scores and their ids
                if hyp['yseq'][-1] == self.eos and i > 2:
                    ys_mask = torch.ones_like(subsequent_mask(len(hyp['yseq'])).unsqueeze(0))
                    ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                else:
                    ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                    ys = torch.tensor(hyp['yseq']).unsqueeze(0)

                if hyp['yseq_asr'][-1] == self.eos and i > 2:
                    ys_mask_asr = torch.ones_like(subsequent_mask(len(hyp['yseq_asr'])).unsqueeze(0))
                    ys_asr = torch.tensor(hyp['yseq_asr']).unsqueeze(0)
                else:
                    ys_mask_asr = subsequent_mask(i + 1).unsqueeze(0)
                    ys_asr = torch.tensor(hyp['yseq_asr']).unsqueeze(0)

                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    if hyp['yseq'][-1] != self.eos or hyp['yseq_asr'][-1] != self.eos or i < 2:
                        local_att_scores, _, local_att_scores_asr, _ = self.dual_decoder.forward_one_step(ys, ys_mask, ys_asr, ys_mask_asr, enc_output,
                                                                                                          cross_mask=None, cross_mask_asr=None,
                                                                                                          cross_self=self.cross_self, cross_src=self.cross_src,
                                                                                                          cross_self_from=self.cross_self_from,
                                                                                                          cross_src_from=self.cross_src_from,
                                                                                                          cross_operator=self.cross_operator, 
                                                                                                          cross_weight=self.cross_weight)
                    if hyp['yseq'][-1] == self.eos and i > 2:
                        local_att_scores = None
                    if hyp['yseq_asr'][-1] == self.eos and i > 2:
                        local_att_scores_asr = None

                if ratio_diverse_st <= 0 and ratio_diverse_asr <=0:
                    if idx == 0:
                        logging.info(f'*** Not force diversity ***')
                    if local_att_scores is not None and local_att_scores_asr is not None:
                        # # Construct score matrix V x V (columns: ST, rows: ASR)
                        # score_matrix = (torch.mm(torch.t(local_att_scores), torch.ones_like(local_att_scores))
                        #                 + torch.mm(torch.t(torch.ones_like(local_att_scores_asr)), local_att_scores_asr)) 
                        # H, W = score_matrix.shape
                        # local_best_scores, indices = score_matrix.view(-1).topk(beam)
                        # local_best_ids = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
                        # local_best_ids_st = local_best_ids[:,0]
                        # local_best_ids_asr = local_best_ids[:,1]

                        x_k, ix_k = local_att_scores.topk(beam)
                        y_k, iy_k = local_att_scores_asr.topk(beam)
                        s_k = (torch.mm(torch.t(x_k), torch.ones_like(x_k))
                                        + torch.mm(torch.t(torch.ones_like(y_k)), y_k))
                        local_best_scores , iv_k = s_k.view(-1).topk(beam)
                        A = torch.LongTensor([[i, j] for i in ix_k.squeeze(0) for j in iy_k.squeeze(0)])
                        ivk = A[iv_k]
                        local_best_ids_st = A[:,0]
                        local_best_ids_asr = A[:,1]
                    elif local_att_scores is not None:
                        local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                        local_best_scores = local_best_scores.squeeze(0)
                        local_best_ids_st = local_best_ids_st.squeeze(0)
                    elif local_att_scores_asr is not None:
                        local_best_scores, local_best_ids_asr = torch.topk(local_att_scores_asr, beam, dim=1)
                        local_best_ids_asr = local_best_ids_asr.squeeze(0)
                        local_best_scores = local_best_scores.squeeze(0)                        
                else:
                    if ratio_diverse_st > 0 and ratio_diverse_asr > 0:
                        if idx == 0:
                            logging.info(f'*** Force diversity for both ST and ASR: not optimized yet ***')
                        if local_att_scores is not None and local_att_scores_asr is not None:
                            count_st = [0] * self.odim
                            count_asr = [0] * self.odim
                            max_count = (1 - ratio_diverse_asr) * beam
                            # Construct score matrix V x V (columns: ST, rows: ASR)
                            score_matrix = (torch.mm(torch.t(local_att_scores), torch.ones_like(local_att_scores))
                                            + torch.mm(torch.t(torch.ones_like(local_att_scores_asr)), local_att_scores_asr))
                            H, W = score_matrix.shape
                            scores_vector, indices = torch.sort(score_matrix.view(-1), descending=True)
                            scores_vector = scores_vector.unsqueeze(1)
                            local_best_ids = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
                            local_best_scores = []
                            local_best_ids_st = []
                            local_best_ids_asr = []
                            for (ii, jj), s in zip(local_best_ids, scores_vector):
                                if count_st[ii] <= max_count and count_asr[jj] <= max_count:
                                    local_best_scores.append(float(s))
                                    local_best_ids_st.append(ii)
                                    local_best_ids_asr.append(jj)
                                    count_st[ii] += 1
                                    count_asr[jj] += 1
                                if len(local_best_scores) >= beam:
                                    break
                        elif local_att_scores is not None:
                            local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                            local_best_scores = local_best_scores.squeeze(0)
                            local_best_ids_st = local_best_ids_st.squeeze(0)
                        elif local_att_scores_asr is not None:
                            local_best_scores, local_best_ids_asr = torch.topk(local_att_scores_asr, beam, dim=1)
                            local_best_scores = local_best_scores.squeeze(0)
                            local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    elif ratio_diverse_asr > 0:
                        max_count = int((1 - ratio_diverse_asr) * beam)
                        if idx == 0:
                            logging.info(f'*** Force diversity for ASR only. max_count = {max_count}, beam = {beam} ***')
                        if local_att_scores is not None and local_att_scores_asr is not None:      
                            # Construct score matrix V x V (rows: ST, columns: ASR)
                            score_matrix = (torch.mm(torch.t(local_att_scores), torch.ones_like(local_att_scores))
                                            + torch.mm(torch.t(torch.ones_like(local_att_scores_asr)), local_att_scores_asr))
                            S, I = torch.topk(score_matrix, dim=0, k=max_count, sorted=False)
                            local_best_scores, indices = torch.topk(S.view(-1), k=beam, sorted=False)
                            local_best_ids_st = I.view(-1)[indices]
                            local_best_ids_asr = indices % self.odim
                        elif local_att_scores is not None:
                            local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1, sorted=False)
                            local_best_scores = local_best_scores.squeeze(0)
                            local_best_ids_st = local_best_ids_st.squeeze(0)
                        elif local_att_scores_asr is not None:
                            local_best_scores, local_best_ids_asr = torch.topk(local_att_scores_asr, beam, dim=1, sorted=False)
                            local_best_scores = local_best_scores.squeeze(0)
                            local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    else:
                        raise NotImplementedError

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[j])
                    if local_att_scores is not None:
                        new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                        new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                        new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids_st[j])
                    else:
                        new_hyp['yseq'] = hyp['yseq']
                    
                    if local_att_scores_asr is not None:
                        new_hyp['yseq_asr'] = [0] * (1 + len(hyp['yseq_asr']))
                        new_hyp['yseq_asr'][:len(hyp['yseq_asr'])] = hyp['yseq_asr']
                        new_hyp['yseq_asr'][len(hyp['yseq_asr'])] = int(local_best_ids_asr[j])
                    else:
                        new_hyp['yseq_asr'] = hyp['yseq_asr']

                    hyps_best_kept.append(new_hyp)

                    hyps_best_kept = sorted(
                        hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))

            if char_list is not None:
                logging.info('best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq']]))
                logging.info('best hypo asr: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq_asr']]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    if hyp['yseq'][-1] != self.eos:
                        hyp['yseq'].append(self.eos)
            if i == maxlen_asr - 1:
                logging.info('adding <eos> in the last postion in the loop for asr')
                for hyp in hyps:
                    if hyp['yseq_asr'][-1] != self.eos:
                        hyp['yseq_asr'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []

            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos and hyp['yseq_asr'][-1] == self.eos:
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

            if char_list is not None:
                for hyp in hyps:
                    logging.info('hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))
                    logging.info('hypo asr: ' + ''.join([char_list[int(x)] for x in hyp['yseq_asr'][1:]]))

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
            return self.recognize_and_translate_sum(x, trans_args, char_list, rnnlm)

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