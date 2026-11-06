import pickle
from baselines.handdgp.third_party.HandMesh.utils import utils


def spiral_tramsform(transform_fp, seq_length, dilation):
    with open(transform_fp, "rb") as f:
        tmp = pickle.load(f, encoding="latin1")
    spiral_indices_list = [
        utils.preprocess_spiral(
            tmp["face"][idx], seq_length[idx], tmp["vertices"][idx], dilation[idx]
        )
        for idx in range(len(tmp["face"]) - 1)
    ]
    return (spiral_indices_list, tmp)
