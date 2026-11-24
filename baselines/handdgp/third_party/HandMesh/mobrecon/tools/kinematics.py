import numpy as np


class MANOHandJoints:
    n_keypoints = 21

    n_joints = 21

    center = 4

    root = 0

    labels = [
        "W",  # 0
        "I0",
        "I1",
        "I2",  # 3
        "M0",
        "M1",
        "M2",  # 6
        "L0",
        "L1",
        "L2",  # 9
        "R0",
        "R1",
        "R2",  # 12
        "T0",
        "T1",
        "T2",  # 15
        "I3",
        "M3",
        "L3",
        "R3",
        "T3",  # 20, tips are manually added (not in MANO)
    ]

    # finger tips are not joints in MANO, we label them on the mesh manually
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}
    # mesh_mapping = {16: 317, 17: 444, 18: 673, 19: 556, 20: 745}

    parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]

    end_points = [0, 16, 17, 18, 19, 20]


class MPIIHandJoints:
    n_keypoints = 21

    n_joints = 21

    center = 9

    root = 0

    labels = [
        "W",  # 0
        "T0",
        "T1",
        "T2",
        "T3",  # 4
        "I0",
        "I1",
        "I2",
        "I3",  # 8
        "M0",
        "M1",
        "M2",
        "M3",  # 12
        "R0",
        "R1",
        "R2",
        "R3",  # 16
        "L0",
        "L1",
        "L2",
        "L3",  # 20
    ]

    parents = [None, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    colors = [
        (0, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (0, 255, 255),
        (0, 255, 255),
        (0, 255, 255),
        (0, 255, 255),
    ]


def mano_to_mpii(mano):
    """
    Map data from MANOHandJoints order to MPIIHandJoints order.

    Parameters
    ----------
    mano : np.ndarray, [21, ...]
      Data in MANOHandJoints order. Note that the joints are along axis 0.

    Returns
    -------
    np.ndarray
      Data in MPIIHandJoints order.
    """
    mpii = []
    for j in range(MPIIHandJoints.n_joints):
        mpii.append(mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])])
    mpii = np.stack(mpii, 0)
    return mpii
