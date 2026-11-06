# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
* @file modules.py
* @author chenxingyu (chenxy.sean@gmail.com)
* @brief Modules composing MobRecon
* @version 0.1
* @date 2022-04-28
*
* @copyright Copyright (c) 2022 chenxingyu
*
"""

import torch.nn as nn


def conv_layer(
    channel_in,
    channel_out,
    ks=1,
    stride=1,
    padding=0,
    dilation=1,
    bias=False,
    bn=True,
    relu=True,
    group=1,
):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [
        _conv(
            channel_in,
            channel_out,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=group,
        )
    ]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0, 4))

    return nn.Sequential(*sequence)
