# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


class Config:
    body_resnet_type = 50
    hand_resnet_type = 50
    trans_resnet_type = 18
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    input_hand_shape = (256, 256)
    input_hm_shape = (64, 64, 64)
    output_body_hm_shape = (8, 64, 48)
    output_hand_hm_shape = (8, 8, 8)
    focal = (5000, 5000)
    princpt = (input_hand_shape[1] / 2, input_hand_shape[0] / 2)
    camera_3d_size = 0.6
    sigma = 2.5


cfg = Config()
