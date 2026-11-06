"""Sequence windows and mesh smoothing from the released OmniHands demo."""

import numpy as np


def gen_sequences(f_num, seq_len, gap):
    sequences = []
    for fid in range(f_num):
        seq = []
        start = fid - (seq_len // 2) * gap
        for f in range(seq_len):
            cur = start + f * gap
            if cur < 0:
                seq.append(0)
            elif cur >= f_num:
                seq.append(f_num - 1)
            else:
                seq.append(cur)
        sequences.append(seq)
    return np.array(sequences)


def seq_smooth(verts_right, verts_left):
    last_r, last_l = None, None
    v_r, v_l, a_r, a_l = None, None, None, None
    last_v_r, last_v_l = None, None

    for i in range(verts_right.shape[0]):
        if last_r is not None and v_r is not None and a_r is not None:
            pred_r = last_r + v_r + 0.5 * a_r
            verts_right[i] = verts_right[i] * 0.7 + pred_r * 0.3

        if last_l is not None and v_l is not None and a_l is not None:
            pred_l = last_l + v_l + 0.5 * a_l
            verts_left[i] = verts_left[i] * 0.7 + pred_l * 0.3

        if last_r is not None:
            v_r = verts_right[i] - last_r

        if last_v_r is not None and v_r is not None:
            a_r = v_r - last_v_r

        if last_l is not None:
            v_l = verts_left[i] - last_l

        if last_v_l is not None and v_l is not None:
            a_l = v_l - last_v_l

        last_r = verts_right[i]
        last_v_r = v_r

        last_l = verts_left[i]
        last_v_l = v_l

    return verts_right, verts_left
