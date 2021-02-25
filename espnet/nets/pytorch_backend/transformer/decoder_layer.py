#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modified by Hang Le
# The original copyright is appended below
# ---
# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""
import logging
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, 
                 self_attn, 
                 src_attn, 
                 feed_forward, 
                 dropout_rate,
                 normalize_before=True, 
                 concat_after=False, 
                 cross_self_attn=None, 
                 cross_src_attn=None, 
                 cross_operator=None, 
                 cross_shared=False,
                 cross_weight_learnable=False, 
                 cross_weight=0.0,
                 adapters=None,
                 adapter_before_src_attn=None,
                 adapter_after_mha=None,
                 ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.adapters = adapters
        self.adapter_before_src_attn = adapter_before_src_attn
        self.adapter_after_mha = adapter_after_mha
        if not cross_shared and cross_self_attn is not None and cross_src_attn is not None:
            self.cross_self_attn = cross_self_attn
            self.cross_src_attn = cross_src_attn
            self.cross_shared = False
        else:
            self.cross_self_attn = None
            self.cross_src_attn = None
            if cross_self_attn is not None:
                self.cross_attn = cross_self_attn
            if cross_src_attn is not None:
                self.cross_attn = cross_src_attn
            if cross_self_attn is None and cross_src_attn is None:
                self.cross_attn = None
            self.cross_shared = True

        self.cross_operator = cross_operator
        if cross_self_attn is not None or cross_src_attn is not None:
            if cross_operator == "concat":
                self.cross_concat_linear1 = nn.Linear(size + size, size)
                self.cross_concat_linear2 = nn.Linear(size + size, size)
            elif cross_operator == "sum":
                if cross_weight_learnable:
                    assert float(cross_weight) > 0
                    self.cross_weight = torch.nn.Parameter(torch.tensor(cross_weight))
                else:
                    self.cross_weight = cross_weight 

        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, 
                memory, memory_mask, 
                cross=None, cross_mask=None, 
                cross_self=False, cross_src=False, 
                lang_id=None, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, 1, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)
            cross (torch.Tensor): decoded previous target from another decoder (batch, max_time_out, size)
        """
        if self.cross_shared:
            cross_self_attn = self.cross_attn
            cross_src_attn = self.cross_attn
        else:
            cross_self_attn = self.cross_self_attn
            cross_src_attn = self.cross_src_attn

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (tgt.shape[0], tgt.shape[1] - 1, self.size), \
                f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
        
        # Self-attention
        if self.concat_after:
            tgt_concat = torch.cat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
            # x = residual + self.concat_linear1(tgt_concat)
            x = self.concat_linear1(tgt_concat)
        else:
            # x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
            x = self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        
        # Cross attention
        if (cross_self_attn is not None and cross is not None and cross_self):
            # mask = torch.einsum('bii,bkk->bik', tgt_q_mask, cross_mask)
            # z = self.dropout(self.cross_attn(tgt_q, cross, cross, mask))
            z = self.dropout(cross_self_attn(tgt_q, cross, cross, cross_mask))
            if self.cross_operator == 'sum':
                x = x + self.cross_weight * z
            elif self.cross_operator == 'concat':
                x = self.cross_concat_linear1(torch.cat((x, z), dim=-1))
            else:
                raise NotImplementedError

        x = x + residual

        if not self.normalize_before:
            x = self.norm1(x)

        # Source attention
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        y = x

        # Adapter before source attention
        if lang_id is not None and self.adapter_before_src_attn is not None:
            x = self.adapter_before_src_attn[lang_id](x, x)[0]
            # x = residual + x

        if self.concat_after:
            x_concat = torch.cat((x, self.src_attn(x, memory, memory, memory_mask)), dim=-1)
            # x = residual + self.concat_linear2(x_concat)
            x = self.concat_linear2(x_concat)
        else:
            # x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
            x = self.dropout(self.src_attn(x, memory, memory, memory_mask))
        
        # Cross attention
        if (cross_src_attn is not None and cross is not None and cross_src):
            # mask = torch.einsum('bii->bi', cross_mask).unsqueeze(1)
            # z = self.dropout(self.cross_attn(y, cross, cross, mask))
            z = self.dropout(cross_src_attn(y, cross, cross, cross_mask))
            if self.cross_operator == 'sum':
                x = x + self.cross_weight * z
            elif self.cross_operator == 'concat':
                x = self.cross_concat_linear2(torch.cat((x, z), dim=-1))
            else:
                raise NotImplementedError
        
        x = x + residual
        
        if not self.normalize_before:
            x = self.norm2(x)
        
        # Adapter after multi-head attention
        if lang_id is not None and self.adapter_after_mha is not None:
            x = self.adapter_after_mha[lang_id](x, x)[0]
            x = x + y
        
        # Feed forward
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        # x = residual + self.dropout(self.feed_forward(x))
        x = self.dropout(self.feed_forward(x))

        # Adapters
        if lang_id is not None and self.adapters is not None:
            x = self.adapters[lang_id](x, x)[0]
        x = residual + x

        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask, cross, cross_mask, cross_self, cross_src
