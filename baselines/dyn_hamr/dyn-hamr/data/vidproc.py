"""Native frame extraction; tracks and cameras come from the adapter's HaMeR and RGB DROID-SLAM stages."""
import os

from ..preproc.extract_frames import split_frame


def is_nonempty(d):
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def preprocess_frames(img_dir, src_path, overwrite=False, **kwargs):
    if not overwrite and is_nonempty(img_dir):
        print(f"FOUND {len(os.listdir(img_dir))} FRAMES in {img_dir}")
        return
    print(f"EXTRACTING FRAMES FROM {src_path} TO {img_dir}")
    print(kwargs)

    # out = video_to_frames(src_path, img_dir, overwrite=overwrite, **kwargs)
    out = split_frame(src_path, img_dir, overwrite=overwrite, **kwargs)
    assert out == 0, "FAILED FRAME EXTRACTION"


def preprocess_tracks(datatype, img_dir, track_dir, shot_dir, gpu, overwrite=False):
    """
    :param img_dir
    :param track_dir, expected format: res_root/track_name/sequence
    :param shot_dir, expected format: res_root/shot_name/sequence
    """
    if not overwrite and is_nonempty(track_dir):
        print(f"FOUND TRACKS IN {track_dir}")
        return

    raise ValueError("Prepared HaMeR tracks are required before optimization")


def preprocess_cameras(cfg, overwrite=False):
    if not overwrite and is_nonempty(cfg.sources.cameras):
        print(f"FOUND CAMERAS IN {cfg.sources.cameras}")
        return
    raise ValueError("Prepared RGB DROID-SLAM cameras are required; the adapter runs the camera stage")
