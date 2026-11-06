import numpy as np
import cv2


def generate_patch_image(
    cvimg, bbox, scale, rot, shift, do_flip, out_shape, shift_wh=None, mask=None
):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
        if mask is not None:
            mask = mask[:, ::-1]

    trans, shift_xy = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        shift,
        shift_wh=shift_wh,
        return_shift=True,
    )
    img_patch = cv2.warpAffine(
        img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR
    )
    img_patch = img_patch.astype(np.float32)
    if mask is not None:
        mask = cv2.warpAffine(
            mask, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR
        )
        mask = (mask > 150).astype(np.uint8)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        shift,
        shift_wh=shift_wh,
        inv=True,
    )

    return img_patch, trans, inv_trans, mask, shift_xy


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(
    c_x,
    c_y,
    src_width,
    src_height,
    dst_width,
    dst_height,
    scale,
    rot,
    shift,
    shift_wh=None,
    inv=False,
    return_shift=False,
):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    if shift_wh is not None:
        shift_lim = (
            max((src_w - shift_wh[0]) / 2, 0),
            max((src_h - shift_wh[1]) / 2, 0),
        )
        x_shift = shift[0] * shift_lim[0]
        y_shift = shift[1] * shift_lim[1]
    else:
        x_shift = y_shift = 0
    src_center = np.array([c_x + x_shift, c_y + y_shift], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    if return_shift:
        return trans, [x_shift / src_w, y_shift / src_h]
    return trans
