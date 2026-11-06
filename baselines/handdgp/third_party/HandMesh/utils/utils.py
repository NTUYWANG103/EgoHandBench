import torch
import numpy as np
import openmesh as om


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals

    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation)
    )
    return spirals
